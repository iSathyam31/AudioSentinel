document.addEventListener('DOMContentLoaded', function() {
    const uploadForm = document.getElementById('uploadForm');
    const audioFileInput = document.getElementById('audioFile');
    const predictButton = document.getElementById('predictButton');
    const predictionResult = document.getElementById('predictionResult');
    const emotionSpan = document.getElementById('emotion');

    uploadForm.addEventListener('submit', function(event) {
        event.preventDefault();
        const formData = new FormData(uploadForm);
        fetch('/predict', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            const predictedEmotion = data.prediction;
            emotionSpan.textContent = predictedEmotion;
            predictionResult.classList.remove('hidden');
        })
        .catch(error => {
            console.error('Error:', error);
        });
    });
});
