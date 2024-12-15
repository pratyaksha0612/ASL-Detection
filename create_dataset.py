import os
import pickle
import mediapipe as mp
import cv2
import math

# Mediapipe configurations
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Initialize Mediapipe Hands
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Directory and Classes
DATA_DIR = './data/ASL'  # Use the ASL folder only
ALPHABETS = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y"]
CLASSES = [f"{letter}_Left" for letter in ALPHABETS] + [f"{letter}_Right" for letter in ALPHABETS]

# Initialize lists for data and labels
data = []
labels = []

def calculate_wrist_angle(landmarks):
    """Calculate the angle of the wrist based on three key points: wrist, index, and pinky."""
    wrist = landmarks[mp_hands.HandLandmark.WRIST]
    index_base = landmarks[mp_hands.HandLandmark.INDEX_FINGER_MCP]
    pinky_base = landmarks[mp_hands.HandLandmark.PINKY_MCP]
    
    # Calculate vectors
    v1 = (index_base.x - wrist.x, index_base.y - wrist.y)
    v2 = (pinky_base.x - wrist.x, pinky_base.y - wrist.y)
    
    # Calculate angle between vectors
    dot_product = v1[0] * v2[0] + v1[1] * v2[1]
    mag_v1 = math.sqrt(v1[0]**2 + v1[1]**2)
    mag_v2 = math.sqrt(v2[0]**2 + v2[1]**2)
    
    if mag_v1 == 0 or mag_v2 == 0:
        return 0.0  # Avoid division by zero
    
    angle = math.acos(dot_product / (mag_v1 * mag_v2))
    return math.degrees(angle)

# Loop through specific classes in the dataset
for class_label in ALPHABETS:  # Use only the alphabet list
    for hand_type in ["Left", "Right"]:
        class_dir = os.path.join(DATA_DIR, f"{class_label}_{hand_type}")
        if not os.path.exists(class_dir):
            print(f"Directory not found: {class_dir}")
            continue

        # Process each image in the class directory
        for img_path in os.listdir(class_dir):
            data_aux = []
            x_ = []
            y_ = []

            img = cv2.imread(os.path.join(class_dir, img_path))
            if img is None:
                print(f"Error reading image: {img_path}")
                continue

            # Convert image to RGB for Mediapipe processing
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            try:
                results = hands.process(img_rgb)
            except Exception as e:
                print(f"Error during hand detection: {e}")
                continue

            if results.multi_hand_landmarks and results.multi_handedness:
                for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                    x_ = [lm.x for lm in hand_landmarks.landmark]
                    y_ = [lm.y for lm in hand_landmarks.landmark]

                    # Normalize landmarks relative to bounding box
                    min_x, max_x = min(x_), max(x_)
                    min_y, max_y = min(y_), max(y_)

                    for lm in hand_landmarks.landmark:
                        data_aux.append((lm.x - min_x) / (max_x - min_x))  # Normalize x
                        data_aux.append((lm.y - min_y) / (max_y - min_y))  # Normalize y
                    
                    # Calculate wrist angle
                    wrist_angle = calculate_wrist_angle(hand_landmarks.landmark)
                    data_aux.append(wrist_angle)  # Add wrist angle

                    # Determine hand type and label
                    detected_hand = handedness.classification[0].label
                    if detected_hand.lower() == hand_type.lower():
                        # Append processed data and labels
                        data.append(data_aux)
                        labels.append(f"{class_label}_{hand_type}")

            print(f"Processed {img_path} for {class_label}_{hand_type}. Data length: {len(data)}, Labels length: {len(labels)}")

# Save data to a pickle file
output_path = 'data_asl.pickle'
print(f"Saving dataset to {output_path}...")
with open(output_path, 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)
print("Dataset saved successfully.")
