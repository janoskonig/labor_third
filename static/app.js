document.addEventListener('DOMContentLoaded', () => {
    const startButton = document.getElementById('startButton');
    const stopButton = document.getElementById('stopButton');
    const severityForm = document.getElementById('severityForm');

    startButton.addEventListener('click', () => {
        fetch('/start_timer', {
            method: 'POST',
        })
        .then(response => response.json())
        .then(data => {
            console.log('Timer started:', data);
            severityForm.style.display = 'block';
        })
        .catch(error => {
            console.error('Error:', error);
        });
    });

    stopButton.addEventListener('click', () => {
        severityForm.style.display = 'none';
    });

    severityForm.addEventListener('submit', (event) => {
        event.preventDefault();
        
        const severity = document.getElementById('severity').value;
        console.log(`Submitting severity: ${severity}`);

        fetch('/end_timer', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ severity }),
        })
        .then(response => response.json())
        .then(data => {
            console.log('Success:', data);
            severityForm.style.display = 'none';
        })
        .catch(error => {
            console.error('Error:', error);
        });
    });
});
