from flask import Flask, render_template, request
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
import numpy as np
import os
import glob
import soundfile
import librosa

app = Flask(__name__)

# Define emotions dictionary if not defined already
emotions = {
    '01': 'neutral',
    '02': 'calm',
    '03': 'happy',
    '04': 'sad',
    '05': 'angry',
    '06': 'fearful',
    '07': 'disgust',
    '08': 'surprised'
}

# Define observed emotions
observed_emotions = ['happy', 'sad', 'angry']

# Modify extract_feature function to include necessary imports
def extract_feature(file_name, mfcc, chroma, mel):
    with soundfile.SoundFile(file_name) as sound_file:
        X = sound_file.read(dtype="float32")
        sample_rate = sound_file.samplerate
        if chroma:
            stft = np.abs(librosa.stft(X))
        result = np.array([])
        if mfcc:
            mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
            result = np.hstack((result, mfccs))
        if chroma:
            chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
            result = np.hstack((result, chroma))
        if mel:
            mel = np.mean(librosa.feature.melspectrogram(y=X, sr=sample_rate).T, axis=0)
            result = np.hstack((result, mel))
    return result

# Modify load_data function to return features and labels
def load_data(test_size=0.2):
    x, y = [], []
    for file in glob.glob("Files/emotion-dataset/Actor_*[0-9]*/*"):
        file_name = os.path.basename(file)
        emotion = emotions[file_name.split("-")[2]]
        if emotion not in observed_emotions:
            continue
        feature = extract_feature(file, mfcc=True, chroma=True, mel=True)
        x.append(feature)
        y.append(emotion)

    return np.array(x), np.array(y)

# Load the dataset
x_train, y_train = load_data()

# Train the model
model1 = MLPClassifier(alpha=0.001, batch_size=128, hidden_layer_sizes=(200, 200, 100, 50), learning_rate='adaptive', max_iter=500)
model1.fit(x_train, y_train)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        audio_file = request.files['file']
        # Save the uploaded file
        audio_file.save("temp.wav")
        # Extract features from the uploaded audio file
        features = extract_feature("temp.wav", mfcc=True, chroma=True, mel=True)
        # Delete the temporary file
        os.remove("temp.wav")
        # Make prediction
        prediction = model1.predict([features])[0]
        return {'prediction': prediction}

if __name__ == '__main__':
    app.run(debug=True)
