import os
import cv2

# Constants
DATA_DIR = './data'
SIGN_LANGUAGES = ["ASL"]  # Collect data for American Sign Language
ALPHABETS = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y"]  
CLASSES = [f"{letter}_Left" for letter in ALPHABETS] + [f"{letter}_Right" for letter in ALPHABETS]  # Add left and right hand variations
DATASET_SIZE = 100  # Number of images per class

# Create directories for the ASL language
for language in SIGN_LANGUAGES:
    language_dir = os.path.join(DATA_DIR, language)
    if not os.path.exists(language_dir):
        os.makedirs(language_dir)
    for label in CLASSES:
        label_dir = os.path.join(language_dir, label)
        if not os.path.exists(label_dir):
            os.makedirs(label_dir)

# Initialize webcam
cap = None
for camera_index in range(5):  # Try up to 5 camera indices
    cap = cv2.VideoCapture(camera_index)
    if cap.isOpened():
        break

if cap is None or not cap.isOpened():
    print("Error: Could not open webcam.")
    exit(1)

def capture_data(language, label):
    """
    Captures all frames for a specific language and label when 'R' is pressed.
    """
    class_dir = os.path.join(DATA_DIR, language, label)
    print(f'Preparing to collect data for {language} - {label}')

    counter = 0
    recording = False  # Flag to start/stop recording frames

    while counter < DATASET_SIZE:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame from webcam.")
            break

        # Fix mirroring by flipping the frame horizontally
        frame = cv2.flip(frame, 1)

        # Display frame with instructions
        # Main label in black
        cv2.putText(frame, f'{language} - {label}: Frame {counter+1}/{DATASET_SIZE}',
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

        # Instructions in red, smaller text
        cv2.putText(frame, 'Press R to start recording, Q to quit',
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        cv2.imshow('frame', cv2.resize(frame, (640, 480)))

        key = cv2.waitKey(25) & 0xFF
        if key == ord('q'):  # Exit if 'Q' is pressed
            print("Recording stopped by user.")
            break
        elif key == ord('r'):  # Start/Stop recording when 'R' is pressed
            if not recording:
                recording = True  # Start recording
                print(f"Started recording for {language} - {label}.")
            else:
                recording = False  # Stop recording
                print(f"Stopped recording for {language} - {label}.")
            # Save frames continuously while recording
        if recording:
            if counter < DATASET_SIZE:
                # Save the frame
                filename = os.path.join(class_dir, f'{counter}.jpg')
                cv2.imwrite(filename, frame)
                counter += 1
                print(f"Captured {counter} image(s) for {language} - {label}.")

    print(f'Finished collecting data for {language} - {label}.')

# Main data collection loop
for language in SIGN_LANGUAGES:  # Only ASL
    for label in CLASSES:  # Both left-hand and right-hand variations
        capture_data(language, label)

# Release resources
cap.release()
cv2.destroyAllWindows()
