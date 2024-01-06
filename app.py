# Import libraries
import streamlit as st
import cv2
import time
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd

# App setup
st.set_page_config(page_title="Emotion Tracker", page_icon=":smiley:", layout="wide")

# Load model
model = tf.keras.models.load_model('models/best_model_facial_meotion_noramlCNN.h5')
face_cascade = cv2.CascadeClassifier('models/haarcascade_frontalface_default.xml')
labels_dict = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Neutral', 5: 'Sad', 6: 'Surprise'}
emotions_history = []

st.title("# Real-time Emotion Recognition")




def detection_face(frame,face_cascade):
    faces = face_cascade.detectMultiScale(frame, scaleFactor=1.3, minNeighbors=3)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 1)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (50, 50, 255), 2)
        cv2.rectangle(frame, (x, y - 40), (x + w, y), (50, 50, 255), -1)
        cv2.putText(frame, "Detecting Face", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    return frame

def detect_face(frame):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_frame)
    return faces[0] if len(faces) == 1 else None

def preprocess(face):
    resized_face = cv2.resize(face, (48, 48))
    normalized_face = resized_face / 255.0
    reshaped_face = np.reshape(normalized_face, (1, 48, 48, 3))
    return reshaped_face

def predict_emotion(preprocessed_face):
    return labels_dict[np.argmax(model.predict(preprocessed_face))]

col1, col2 = st.columns(2)
storing = False
with col1:
    run = st.checkbox('Run')
    FRAME_WINDOW = st.image([])
    camera = cv2.VideoCapture(0)

    start_time = time.time()
    prediction_interval = 5
    total_time = 0
    
    while run and total_time < 60  :  
        emotion_text = st.empty() 
        emotion_text.empty()
        _, frame = camera.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.flip(frame, 1) 
        frame = detection_face(frame, face_cascade)
        FRAME_WINDOW.image(frame)
        
        if frame is not None:
            frame_BGR = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            face = detect_face(frame_BGR)

            elapsed_time = time.time() - start_time
            if elapsed_time >= prediction_interval:
                if face is not None:
                    preprocessed_face = preprocess(frame_BGR[face[1]:face[1]+face[3], face[0]:face[0]+face[2]])
                    emotion = predict_emotion(preprocessed_face)
                    emotion_text.text(f"Emotion: {emotion}")
                    emotions_history.append(emotion)
                else:
                    emotion_text.text(f"No prediction because of Face not detecting")
                
                start_time = time.time() 
                total_time += elapsed_time  

    camera.release()
    FRAME_WINDOW.empty()
    if total_time > 60:

        storing=True
with col2:

    
    # Charts 
    charts = []
    emotion_colors = {
    'Angry': 'red',
    'Disgust': 'green',
    'Fear': 'purple',
    'Happy': 'yellow',
    'Neutral': 'gray',
    'Sad': 'blue',
    'Surprise': 'orange'
    }
    emotions_history = ['Happy', 'Sad', 'Happy', 'Neutral', 'Angry', 'Sad', 'Happy', 'Surprise', 'Neutral']
    st.markdown("""---""")
    
    st.markdown("""### How it works:
     - Face the webcam to detect emotions
     - Emotions display on screen  
     - Analyze trends with charts""")
    if storing:
        col3, col4 = st.columns(2)

        with col3:
            history_df = pd.DataFrame(emotions_history, columns=['emotion']) 

            # Get counts per emotion
            emotion_counts = history_df['emotion'].value_counts()
            
            # Plot 3D bar chart
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')

            x = emotion_counts.index
            y = np.arange(len(x))
            z = emotion_counts.values
            colors = [emotion_colors[label] for label in x]
            ax.bar(x, z, zs=y, zdir='y', color=colors, alpha=0.8)

            ax.set_xlabel('Emotions')
            ax.set_ylabel('Frequency')
            ax.set_zlabel('Count')

            st.pyplot(fig)
        with col4:
            emotion_count = history_df['emotion'].value_counts()
            fig, ax = plt.subplots()
            colors = [emotion_colors[label] for label in emotion_count.index]
            ax.pie(emotion_count, labels=emotion_count.index, autopct='%1.1f%%', colors=colors)
            ax.set_title('Emotion Distribution')

            st.pyplot(fig)

# Page: Analysis  
st.title("# Emotional Data Analysis")

st.markdown("""
    Emotion Trends Over Time:
    Analyzing the emotional data collected over the session reveals intriguing trends.
    The 3D bar chart showcases the frequency of each emotion, providing insights into the emotional landscape during the interaction.
""")

# Distribution of Emotions
st.markdown("""
    Distribution of Emotions:
    The pie chart illustrates the distribution of emotions detected, offering a clear visualization of the prevalence of each emotional state.
    This breakdown enhances our understanding of the user's emotional experiences.
""")

for chart in charts:
   st.download_button('Download', chart, file_name='chart.png') 
   st.pyplot(chart)
   st.markdown("...")
   
# Page: About
st.title("# About") 
st.markdown("""
    Project Overview:
    The "Emotion Tracker" is a real-time emotion recognition application designed by John Doe.
    It utilizes facial emotion recognition technology to detect and analyze emotions during live webcam interactions.
""")

# Purpose and Inspiration
st.markdown("""
    Purpose and Inspiration:
    Inspired by the curiosity to explore the world of emotions, this project aims to provide a unique perspective
    on how individuals express themselves through facial cues. The goal is to create an engaging and insightful experience for users.
""")


st.markdown("""---""")
st.markdown("Made by Min Thaw Phyo(Justin)")
st.markdown('My GitHub Profile: [Minthawphyo](https://github.com/Minthawphyo)')
st.markdown("""

    This application was crafted with passion and a keen interest in the intersection of technology and human emotions.
    Feel free to explore, analyze, and share your thoughts on the emotional journey captured by the "Emotion Tracker."
""")
