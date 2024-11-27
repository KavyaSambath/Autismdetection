import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import librosa
import librosa.display
import pyaudio
import wave
from flask import Flask, render_template, request, redirect, url_for, jsonify
from werkzeug.utils import secure_filename

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'wav', 'jpg', 'png'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
face_model = tf.keras.models.load_model("models/FACE.model")
sound_model = tf.keras.models.load_model("models/SOUND.model")
CATEGORIES = ["AUTISM", "NORMAL"]

class conf:
    sampling_rate = 44100
    duration = 3
    hop_length = 700 * duration
    fmin = 1
    fmax = sampling_rate // 2
    n_mels = 256
    n_fft = n_mels * 20
    samples = sampling_rate * duration

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def prepare_face(file):
    IMG_SIZE = 50
    img_array = cv2.imread(file, 1)
    img_array = cv2.medianBlur(img_array, 1)
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    new_array = np.expand_dims(new_array, axis=0)
    return new_array

def prepare_sound(file):
    IMG_SIZE = 50
    img_array = cv2.imread(file, 1)
    img_array = cv2.Canny(img_array, threshold1=10, threshold2=10)
    img_array = cv2.medianBlur(img_array, 1)
    img_array = cv2.equalizeHist(img_array)
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    new_array = np.expand_dims(new_array, axis=0)
    return new_array

def read_audio(conf, pathname, trim_long_data):
    y, sr = librosa.load(pathname, sr=conf.sampling_rate)
    if 0 < len(y):
        y, _ = librosa.effects.trim(y)
    if len(y) > conf.samples:
        if trim_long_data:
            y = y[0:0 + conf.samples]
    else:
        padding = conf.samples - len(y)
        offset = padding // 2
        y = np.pad(y, (offset, conf.samples - len(y) - offset), 'constant')
    return y

def audio_to_melspectrogram(conf, audio):
    spectrogram = librosa.feature.melspectrogram(audio, sr=conf.sampling_rate, n_mels=conf.n_mels, hop_length=conf.hop_length, n_fft=conf.n_fft, fmin=conf.fmin, fmax=conf.fmax)
    spectrogram = librosa.power_to_db(spectrogram)
    spectrogram = spectrogram.astype(np.float32)
    return spectrogram

def read_as_melspectrogram(conf, pathname, trim_long_data, debug_display=False):
    x = read_audio(conf, pathname, trim_long_data)
    mels = audio_to_melspectrogram(conf, x)
    return mels

def save_image_from_sound(img_path):
    x = read_as_melspectrogram(conf, img_path, trim_long_data=False, debug_display=True)
    plt.imshow(x, interpolation='nearest')
    plt.savefig("uploads/1.png")
    plt.close()

def capture_face():
    cam = cv2.VideoCapture(0)
    while True:
        ret, img = cam.read()
        img = cv2.flip(img, 1)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w + 50, y + h + 50), (255, 0, 0), 2)
            im = gray[y:y + h, x:x + w]
            cv2.imshow('image', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.imwrite("uploads/main.jpg", im)
            break
    cam.release()
    cv2.destroyAllWindows()

def record_audio(filename="uploads/output.wav", duration=2):
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 44100

    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)

    print("Recording...")
    frames = []
    for i in range(0, int(RATE / CHUNK * duration)):
        data = stream.read(CHUNK)
        frames.append(data)
    print("Finished recording")

    stream.stop_stream()
    stream.close()
    p.terminate()

    wf = wave.open(filename, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        save_image_from_sound(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        capture_face()
        face_prediction = face_model.predict(prepare_face("uploads/main.jpg"))
        sound_prediction = sound_model.predict(prepare_sound("uploads/1.png"))
        face_prediction = list(face_prediction[0])
        sound_prediction = list(sound_prediction[0])
        final_prediction = [(face + sound) / 2 for face, sound in zip(face_prediction, sound_prediction)]
        final_result = CATEGORIES[final_prediction.index(max(final_prediction))]
        return jsonify(result=final_result)
    return redirect(request.url)

@app.route('/record', methods=['POST'])
def record_and_predict():
    record_audio()
    save_image_from_sound("uploads/output.wav")
    capture_face()
    face_prediction = face_model.predict(prepare_face("uploads/main.jpg"))
    sound_prediction = sound_model.predict(prepare_sound("uploads/1.png"))
    face_prediction = list(face_prediction[0])
    sound_prediction = list(sound_prediction[0])
    final_prediction = [(face + sound) / 2 for face, sound in zip(face_prediction, sound_prediction)]
    final_result = CATEGORIES[final_prediction.index(max(final_prediction))]
    return jsonify(result=final_result)

if __name__ == "__main__":
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    app.run(debug=True)
