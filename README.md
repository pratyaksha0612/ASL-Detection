# ASL-Detection System

This project focuses on building a real-time Sign Language Detection System using computer vision and machine learning techniques. The system accurately recognizes and interprets American Sign Language (ASL) gestures, bridging the communication gap between sign language users and non-users.
By leveraging MediaPipe, OpenCV, and scikit-learn libraries, the project demonstrates the power of machine learning and computer vision to create an accessible, user-friendly solution. The pipeline integrates data collection, preprocessing, model training, and real-time gesture recognition.

**Technologies Used**:
1. _OpenCV:_ Image processing and video stream handling.
2. _MediaPipe Hands_: Hand landmark detection and feature extraction.
3. _scikit-learn:_ Model training, evaluation, and deployment.
4. _Random Forest Classifier:_ Supervised machine learning algorithm for robust and accurate classification.
5. _Python:_ The core programming language for all components.

**System Workflow**
1. _Data Collection:_ Images of ASL gestures are captured and labeled.
2. _Preprocessing:_ Extract hand landmarks, normalize data, and compute additional features (wrist angle).
3. _Model Training:_ Train the Random Forest Classifier on preprocessed data and save the model.
4._ Real-Time Prediction:_ Feed webcam video frames into the model to predict gestures in real-time.
5. _Visualization:_ Display bounding boxes and gesture labels on the video stream.

DATA COLLECTION:
![2](https://github.com/user-attachments/assets/8c667b62-ebd1-4db6-a6d2-5bcf96357101)

DATASET LANDMARKS:
![A_Right_Right](https://github.com/user-attachments/assets/56569088-d515-4181-83d4-1ffb2465e9a8)

REAL-TIME GESTURE DETECTION:
![Screenshot 2024-12-17 223620](https://github.com/user-attachments/assets/73a9d74c-e78f-4cb0-abb0-8078c46755c2)
