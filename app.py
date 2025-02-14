from flask import Flask, render_template, Response, jsonify, request
import cv2
import classifier  # Import the provided classifier
import pyttsx3  # Text-to-speech
import time
import threading
from googletrans import Translator  # For language translation

app = Flask(__name__)

cap = cv2.VideoCapture(0)  # Open webcam
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Reduce resolution for better performance
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

prediction_text = "....."
translator = Translator()  # Initialize translator
last_prediction_time = 0  # Track last prediction time
selected_language = "en"  # Default language is English

# Function to translate text (only when needed)
def translate_text(text, target_lang):
    if target_lang == "en":  # No need to translate English
        return text
    try:
        translated = translator.translate(text, dest=target_lang)
        return translated.text
    except:
        return text  # If translation fails, return original text

# Function to speak only the first letter or digit in a separate thread
def speak_text(text):
    if text.strip():  # Ensure text is not empty
        translated_text = translate_text(text, selected_language)  # Translate full text
        tts_thread = threading.Thread(target=speak, args=(translated_text,))
        tts_thread.start()

def speak(text):
    engine = pyttsx3.init()  # Create a new engine instance in the thread
    voices = engine.getProperty('voices')
    for voice in voices:
        if "female" in voice.name.lower():  
            engine.setProperty('voice', voice.id)
            break
    engine.say(text)
    engine.runAndWait()

def generate_frames():
    global prediction_text, last_prediction_time
    while True:
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)  # Mirror the frame

        current_time = time.time()
        if current_time - last_prediction_time > 0.5:  # Limit to 2 predictions per second
            last_prediction_time = current_time
            frame, prediction_text = classifier.get_prediction(frame)
            
            # Speak prediction only when it changes
            speak_text(prediction_text)

        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/predict')
def predict():
    global prediction_text
    return jsonify({'prediction': prediction_text})

@app.route('/change_language', methods=['POST'])
def change_language():
    global selected_language
    selected_language = request.json.get("language", "en")  # Default to English
    return jsonify({'message': f'Language changed to {selected_language}'})

if __name__ == "__main__":
    app.run(debug=True)
