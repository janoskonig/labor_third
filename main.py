import os
from flask import Flask, render_template, request, jsonify, redirect
from dotenv import load_dotenv
import pytz
import matplotlib
import matplotlib.colors as mcolors
import time
import threading
from datetime import datetime
import base64
from io import BytesIO
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import mysql.connector
from mysql.connector.pooling import MySQLConnectionPool

matplotlib.use('Agg')
import matplotlib.pyplot as plt



app = Flask(__name__)

# Set your local timezone and the server timezone
LOCAL_TIMEZONE = pytz.timezone("Europe/Budapest")  # Example: Budapest
SERVER_TIMEZONE = pytz.timezone("UTC")  # Assuming server is in UTC

def get_local_time():
    # Get the current time in the server's timezone and convert to local timezone
    server_time = datetime.now(SERVER_TIMEZONE)
    local_time = server_time.astimezone(LOCAL_TIMEZONE)
    return local_time

load_dotenv(".env")
# Retrieve environment variables
host = os.getenv("DB_HOST")
port = os.getenv("DB_PORT")
user = os.getenv("DB_USER")
password = os.getenv("DB_PASSWORD")
database = os.getenv("DB_NAME")
# Ensure all variables are retrieved correctly
if not all([host, port, user, password, database]):
    raise ValueError("Missing one or more environment variables")

# Initialize the connection pool
pool = MySQLConnectionPool(
    pool_name="my_pool",
    pool_size=10,  # Adjust size according to your application's requirement
    host=os.getenv("DB_HOST"),
    port=os.getenv("DB_PORT"),
    user=os.getenv("DB_USER"),
    password=os.getenv("DB_PASSWORD"),
    database=os.getenv("DB_NAME")
)

def get_db_connection():
    return pool.get_connection()


def get_db_cursor():
    conn = get_db_connection()
    return conn.cursor()

# Save contraction to database
def save_contraction_to_db(start_time, end_time, duration, severity):
    conn = get_db_connection()
    try:
        print(f"Inserting data into DB: start={start_time}, end={end_time}, duration={duration}, severity={severity}")  # Debug
        with conn.cursor() as cursor:
            sql = "INSERT INTO igazidata (start_time, end_time, duration, severity) VALUES (%s, %s, %s, %s)"
            values = (start_time, end_time, duration, severity)
            cursor.execute(sql, values)
        conn.commit()
        print("Data inserted successfully")  # Debug
    except mysql.connector.Error as e:
        print(f"Error inserting data: {e}")  # Debug
    finally:
        conn.close()


# Fetch contractions from database
def fetch_contractions_from_db():
    try:
        conn = get_db_connection()
        with conn.cursor() as cursor:
            cursor.execute("SELECT * FROM igazidata")
            data = cursor.fetchall()
            df = pd.DataFrame(data, columns=["id", "start_time", "end_time", "duration", "severity"])
            print(f"Fetched data columns: {df.columns}")  # Debug
        return df
    except Exception as e:
        print(f"Error fetching data: {e}")  # Debug
        return pd.DataFrame()
    finally:
        conn.close()
    

# Plotting function
def plot_to_base64(fig):
    buf = BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()
    return img_str


# Prediction and plotting function
def analyze_contractions(data, window_size=5, exclude_fraction=1/4, future_pred_length=36000):
    """
    Analyzes contraction durations using rolling statistics and fits an exponential decay model
    to predict future contractions and the estimated time of childbirth.

    Parameters:
        data (pd.DataFrame): The input dataframe containing 'start_time' and 'duration' columns.
        window_size (int): The window size for rolling calculations.
        exclude_fraction (float): The fraction of initial data to exclude from exponential decay fitting.
        future_pred_length (int): The number of future time points to predict.

    Returns:
        popt (tuple): The optimal parameters for the exponential decay model.
        childbirth_timepoint (float): The predicted time point of childbirth.
    """

    # Remember the first start_time:
    if data.empty:
        return None, None, None, None
    else:
        veryfirst_start_time = data['start_time'].iloc[0]
    
    # Converting start_time to passed time in seconds
    data['start_time'] = (data['start_time'] - data['start_time'].min()).dt.total_seconds()
    # Calculate rolling standard deviation
    rolling_std = data['duration'].rolling(window=window_size).std()

    # Fill NaNs with appropriate values
    rolling_std_filled = rolling_std.bfill()

    # Prepare time series for rolling std
    std_time = data['start_time']

    # Exclude the first part of the rolling standard deviation data
    start_index = int(len(rolling_std_filled) * exclude_fraction)
    filtered_std_time = std_time[start_index:]
    filtered_rolling_std = rolling_std_filled[start_index:]

    # Exponential decay function
    def exp_decay(x, a, b, c):
        return a * np.exp(-b * x) + c

    # Fit the exponential decay function
    popt, pcov = curve_fit(exp_decay, filtered_std_time, filtered_rolling_std, p0=(1, 0.001, 0), maxfev=10000)

    # Predict rolling standard deviation
    rolling_std_pred = exp_decay(std_time, *popt)

    # Extend future predictions
    future_times = np.linspace(std_time.max() + 1, std_time.max() + future_pred_length, num=future_pred_length)
    future_rolling_std_pred = exp_decay(future_times, *popt)
    # Find the timepoint where rolling_std becomes 0
    childbirth_timepoint = future_times[np.argmin(np.abs(future_rolling_std_pred))]

    # Convert childbirth_timepoint to datetime
    childbirth_timepoint = pd.to_datetime(veryfirst_start_time + pd.to_timedelta(childbirth_timepoint, unit='s'))
    # Convert future times to datetime
    future_times = pd.to_datetime(veryfirst_start_time + pd.to_timedelta(future_times, unit='s'))
    
    # Convert std_time to datetime
    std_time = pd.to_datetime(data['start_time'], unit='s')
    data["start_time"] = veryfirst_start_time + pd.to_timedelta(data["start_time"], unit='s')
    # Plot rolling standard deviation with exponential decay fit
    fig1 = plt.figure(figsize=(14, 7))
    plt.plot(data["start_time"], rolling_std_filled, label='Csúszóátlag szórása', color='blue')
    plt.plot(data["start_time"], rolling_std_pred, color='green', linewidth=2, label='Illesztett exponenciális görbe')
    plt.plot(future_times, future_rolling_std_pred, label='Előrejelzés', color='purple')
    plt.axvline(x=childbirth_timepoint, color='red', linestyle='--', label='Születés várható ideje')
    plt.xlabel('Időpont')
    plt.ylabel('Csúszóátlag szórása')
    plt.title(f'Csúszóátlag szórása exponenciális illesztéssel, {window_size}-es ablakmérettel')
    plt.legend()
    # plt.show()

    # Calculate rolling mean and 2 SD brackets for existing data
    rolling_mean = data['duration'].rolling(window=window_size).mean()
    rolling_std = data['duration'].rolling(window=window_size).std()
    rolling_upper_bound = rolling_mean + (2 * rolling_std)
    rolling_lower_bound = rolling_mean - (2 * rolling_std)

    # Fill NaNs with appropriate values
    rolling_lower_bound_filled = rolling_lower_bound.bfill()
    rolling_upper_bound_filled = rolling_upper_bound.bfill()

    # Convert rolling bounds to numpy arrays
    rolling_lower_bound_array = np.array(rolling_lower_bound_filled, dtype=float)
    rolling_upper_bound_array = np.array(rolling_upper_bound_filled, dtype=float)

    # Plot contraction durations with rolling statistics and predicted childbirth
    fig2 = plt.figure(figsize=(12, 6))
    plt.scatter(data['start_time'], data['duration'], label='Összehúzódások időtartama')
    plt.plot(data['start_time'], rolling_mean, color='r', linestyle='-', label='Csúszóátlag')
    plt.plot(data['start_time'], rolling_lower_bound_array, color='g', linestyle='--', label=' ± 2 SD')
    plt.plot(data['start_time'], rolling_upper_bound_array, color='g', linestyle='--')
    plt.fill_between(data['start_time'], rolling_lower_bound_array, rolling_upper_bound_array, color='g', alpha=0.1)
    plt.axvline(x=childbirth_timepoint, color='red', linestyle='--', label='Születés várható ideje')
    plt.xlabel('Időpont (kontrakció kezdete)')
    plt.ylabel('Időtartam (másodperc)')
    plt.title(f'Összehúzúdások időtartama, csúszóátlag ± 2 SD (ablakméret: {window_size})')
    plt.legend()
    # plt.show()

    return popt, childbirth_timepoint, fig1, fig2

@app.route('/')
def index():
    # Usage
    data = fetch_contractions_from_db()
    if data.empty:
        return render_template('index.html', popt=None, img_str1=None, img_str2=None, childbirth_timepoint=None, table=None)
    # print(data)
    popt, chilbirth_timepoint, fig1, fig2 = analyze_contractions(data)
    img_str1 = plot_to_base64(fig1)
    img_str2 = plot_to_base64(fig2)
    return render_template('index.html', popt=popt, img_str1=img_str1, img_str2=img_str2, childbirth_timepoint=chilbirth_timepoint, table=data.to_html())


start_time = None
@app.route('/start_timer', methods=['POST'])
def start_timer():
    global start_time
    start_time = get_local_time()
    print(f"Timer started at {start_time}")
    return jsonify({"start_time": start_time.strftime("%Y-%m-%d %H:%M:%S")})

@app.route('/end_timer', methods=['POST'])
def end_timer():
    global start_time
    if start_time is None:
        print("Timer has not been started")
        return jsonify({"error": "Timer has not been started"}), 400
    end_time = get_local_time()
    # Convert start_time and end_time to seconds to allow for duration calculation

    duration = (end_time - start_time).total_seconds()
    print(f"Timer ended at {end_time}")


    print(f"Calculated duration: {duration} seconds")  # Debug

    severity = request.json.get('severity')
    print(f"Received severity: {severity}")  # Debug

    save_contraction_to_db(start_time, end_time, duration, severity)
    start_time = None
    # return end_time, duration, severity
    return jsonify({"end_time": end_time.strftime("%Y-%m-%d %H:%M:%S"), "duration": duration, "severity": severity})

    

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5050, debug=False)