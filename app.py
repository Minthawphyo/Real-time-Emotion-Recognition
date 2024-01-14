import threading
import streamlit as st
import cv2
import time
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer
import base64

# Thread safety
lock = threading.Lock()

# Containers for image and emotions
img_container = {"img": None}
emotional_container = {"emotions": None}

# Page configuration
st.set_page_config(page_title="Emotion Tracker", page_icon=":smiley:", layout="wide")

# Load emotion recognition model and face cascade classifier
model = tf.keras.models.load_model('models/best_model_facial_meotion_noramlCNN.h5')
face_cascade = cv2.CascadeClassifier('models/haarcascade_frontalface_default.xml')

# Dictionary mapping emotion labels to indices
labels_dict = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Neutral', 5: 'Sad', 6: 'Surprise'}

# Title for the application
st.title("Emotion Tracker")

# Sidebar for navigation
page = st.sidebar.radio("Navigate", ["Live Analysis", "Historical Data", "About"])

# Function to draw rectangles around detected faces
def detection_face(frame, face_cascade):
    # Detect faces using the cascade classifier
    faces = face_cascade.detectMultiScale(frame, scaleFactor=1.3, minNeighbors=3)
    
    # Draw rectangles around detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 1)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (50, 50, 255), 2)
        cv2.rectangle(frame, (x, y - 40), (x + w, y), (50, 50, 255), -1)
        cv2.putText(frame, "Detecting Face", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    return frame

# Function to detect a single face
def detect_face(frame):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_frame)
    return faces[0] if len(faces) == 1 else None

# Function to preprocess a face for emotion prediction
def preprocess(face):
    resized_face = cv2.resize(face, (48, 48))
    normalized_face = resized_face / 255.0
    reshaped_face = np.reshape(normalized_face, (1, 48, 48, 3))
    return reshaped_face

# Function to predict emotion from a preprocessed face
def predict_emotion(preprocessed_face):
    return labels_dict[np.argmax(model.predict(preprocessed_face))]

# Callback function for processing video frames
def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")
    
    # Update the image container with the latest frame
    with lock:
        img_container["img"] = img
    
    return frame


def get_table_download_link(df):
    """Generates a link allowing the data in a given panda dataframe to be downloaded
    in:  dataframe
    out: href string
    """
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  
    href = f'<a href="data:file/csv;base64,{b64}">Download csv file</a>'


if 'history_df' not in st.session_state:
    st.session_state.history_df = pd.DataFrame(columns=['emotion'])

if 'submitted' not in st.session_state:
    st.session_state.submitted = False

# Live Analysis Page
if page == "Live Analysis":
    # Initialization
    start_time = None
    prediction = 0
    emotions_history = []
    prediction_interval = 5
    total_time = 0
    history_df = None

    # Streamlit layout setup
    col1, col2 = st.columns(2)
    with col1:
        ctx = webrtc_streamer(key="example", video_frame_callback=video_frame_callback, media_stream_constraints={"video": True, "audio": False})
    
    with col2:
        fig, ax = plt.subplots(figsize=(10, 6))
        emotion_colors = {
            'Angry': 'red',
            'Disgust': 'green',
            'Fear': 'purple',
            'Happy': 'yellow',
            'Neutral': 'gray',
            'Sad': 'blue',
            'Surprise': 'orange'
        }
        st.markdown("---")
        st.markdown("### Live Analysis")
        st.markdown("Face the webcam to detect emotions in real-time. Emotions will be displayed on the screen, and trends can be analyzed using charts.")

        # Placeholder for the emotion chart
        fig_place = st.empty()

    # Main loop for real-time processing
    while ctx.state.playing:
        if start_time is None:
            start_time = time.time()
        
        with lock:
            img = img_container["img"]
        
        if img is None:
            continue
        
        frame_BGR = cv2.flip(img, 1)
        frame_BGR = detection_face(frame_BGR, face_cascade)

        if frame_BGR is not None:
            frame_BGR = cv2.cvtColor(frame_BGR, cv2.COLOR_RGB2BGR)
            
            face = detect_face(frame_BGR)
            elapsed_time = time.time() - start_time
            
            print(elapsed_time)
            print("The prediction", (prediction))
            print("The history", (emotions_history))
            
            if elapsed_time >= prediction_interval and prediction < 5:
                if face is not None:
                    preprocessed_face = preprocess(frame_BGR[face[1]:face[1] + face[3], face[0]:face[0] + face[2]])
                    emotion = predict_emotion(preprocessed_face)
                    print(emotion)
                    
                    emotions_history.append(emotion)
                    history_df = pd.DataFrame(emotions_history, columns=['emotion'])
                    st.session_state.history_df=pd.DataFrame(emotions_history, columns=['emotion'])
                    st.session_state.submitted=True
                    prediction += 1

                start_time = time.time()
                total_time += elapsed_time
            
            if history_df is not None:
                emotion_counts = history_df['emotion'].value_counts()

                x = emotion_counts.index
                y = np.arange(len(x))
                z = emotion_counts.values
                colors = [emotion_colors[label] for label in x]
                ax.cla()
                ax.bar(x, z, color=colors, alpha=0.8)

                ax.set_xlabel('Emotions')
                ax.set_ylabel('Frequency')
                ax.set_title('Emotion Frequency')
                
                # Display the emotion chart
                fig_place.pyplot(fig)
    
    # Historical Data Page
elif page == "Historical Data":

    st.title("Historical Data Analysis")
    history_df = st.session_state.history_df
    st.write(st.session_state.history_df)
    # Placeholder for historical data analysis content
    st.markdown("""
        Emotion Trends Over Time:
        Analyzing the emotional data collected over the session reveals intriguing trends.
        The 3D bar chart showcases the frequency of each emotion, providing insights into the emotional landscape during the interaction.
    """)

    # Distribution of Emotions
    st.markdown("""
        Distribution of Emotions:
        The bar chart illustrates the distribution of emotions detected, offering a clear visualization of the prevalence of each emotional state.
        This breakdown enhances our understanding of the user's emotional experiences.
    """)
    


    # Historical data analysis chart placeholders
    fig_hist_trends, ax_hist_trends = plt.subplots(figsize=(10, 6))
    fig_hist_distribution, ax_hist_distribution = plt.subplots(figsize=(10, 6))
    
    
    # Placeholder for the historical trends chart
    fig_place_hist_trends = st.empty()

    # Placeholder for the historical distribution chart
    fig_place_hist_distribution = st.empty()

    emotion_mapping = {'Angry': 0, 'Disgust': 1, 'Fear': 2, 'Happy': 3, 'Neutral': 4, 'Sad': 5, 'Surprise': 6}
    history_df['emotion_numeric'] = history_df['emotion'].map(emotion_mapping)

    ax_hist_trends.plot(history_df.index, history_df['emotion_numeric'], marker='o', linestyle='-')
    ax_hist_trends.set_xlabel('Time')
    ax_hist_trends.set_ylabel('Emotion')
    ax_hist_trends.set_title('Historical Trends Over Time')


    fig_hist_distribution, ax_hist_distribution = plt.subplots(figsize=(10, 6))
    emotion_counts = history_df['emotion'].value_counts()
    ax_hist_distribution.bar(emotion_counts.index, emotion_counts.values, color='skyblue')
    ax_hist_distribution.set_xlabel('Emotion')
    ax_hist_distribution.set_ylabel('Count')
    ax_hist_distribution.set_title('Distribution of Emotions')







    # Main loop for historical data analysis
    if  st.session_state.submitted:
        # Placeholder for the historical trends chart
        fig_place_hist_trends.pyplot(fig_hist_trends)

        # Placeholder for the historical distribution chart
        fig_place_hist_distribution.pyplot(fig_hist_distribution)

          

        

        # st.markdown(get_table_download_link(history_df), unsafe_allow_html=True)
        # # Provide download link for CSV export

    # About Page
elif page == "About":
    st.title("About")

    # Project Overview
    st.markdown("""
        Project Overview:
        The "Emotion Tracker" is a real-time emotion recognition application designed by John Doe.
        It utilizes facial emotion recognition technology to detect and analyze emotions during live webcam interactions.
    """)


    st.markdown("""
        Purpose and Inspiration:
        Inspired by the curiosity to explore the world of emotions, this project aims to provide a unique perspective
        on how individuals express themselves through facial cues. The goal is to create an engaging and insightful experience for users.
    """)

    
    st.image("./images/profile.jpg", caption="Developer",width=200)

    # GitHub link
    st.markdown("""
        [Source Code on GitHub](https://github.com/Minthawphyo/EmotionTracker) 
        (Feel free to explore and contribute!)
    """)


    st.markdown("""
        Contact Information:
        For inquiries or feedback, please contact us at [dividechange@gmail.com](mailto:email@example.com).
    """)
