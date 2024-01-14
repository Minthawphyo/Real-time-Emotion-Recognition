# Emotion Tracker App

## Overview
The Emotion Tracker App is a real-time webcam-based application that utilizes facial recognition to detect emotions and analyze emotional trends over time. This app is designed to provide insights into emotional frequencies and distribution through charts.

## Usage
1. Clone the repository.
2. Run the following command to start the app:
    ```bash
    streamlit run app.py
    ```
3. Allow webcam access when prompted.
4. Face the webcam to detect emotions. Emotions will be displayed on the screen as they are recognized.
5. Charts are generated to visualize emotion frequencies and distribution.

## Models
The `models` folder contains the following:

- `best_model_facial_emotion_normalCNN.h5`: Trained Keras CNN model for facial emotion recognition.
- `haarcascade_frontalface_default.xml`: Haar cascade classifier for face detection.

The model was trained on the FER-2022 emotion classification dataset. Refer to the `notebooks` directory for details on model training and testing.

## Notebooks
- `model_training.ipynb`: Notebook for training the emotion recognition model.
- `model_testing.ipynb`: Notebook for testing the model on real-time webcam video.

## Requirements
Install the necessary dependencies using the following command:
```bash
pip install -r requirements.txt
```
## Main Dependencies:

- OpenCV
- TensorFlow
- Streamlit
- Matplotlib
- Pandas
- streamlit
- pandas
- numpy
- opencv-python-headless
- scikit-learn
- pillow
- streamlit_webrtc 

## Credits

- FER-2022 dataset from Kaggle
- Haar cascade classifier from OpenCV

Feel free to explore and contribute to the project!
