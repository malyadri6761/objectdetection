import streamlit as st
import streamlit.components.v1 as components
import base64
import os
import cv2
import tempfile
import numpy as np
from PIL import Image
from prediction import YOLO_Pred
import tempfile
import numpy as np
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
import av
yolo=YOLO_Pred('./best.onnx','./data.yaml')

st.set_page_config(layout="wide")

# Title
st.title("My App")

# Navigation bar simulated using columns
nav1, nav2, nav3, nav4 = st.columns(4)

# Placeholder to store current section
if 'section' not in st.session_state:
    st.session_state.section = 'Home'

# Navigation buttons
with nav1:
    if st.button("🏠 Home"):
        st.session_state.section = 'Home'

with nav2:
    if st.button("📄 Description"):
        st.session_state.section = 'Description'

with nav3:
    if st.button("🖼️ Gallery"):
        st.session_state.section = 'Gallery'

with nav4:
    if st.button("🤖 Model"):
        st.session_state.section = 'Model'

# Display section content
st.markdown("---")
if st.session_state.section == 'Home':
    st.markdown("""
<h2 style="color:#1f77b4;">📌 Object Detection Process – Step-by-Step Summary</h2>

<h4>1. Dataset Selection</h4>
<p>I began by selecting the <strong>Pascal VOC 2012</strong> dataset from Kaggle, a well-structured dataset widely used for object detection tasks. It contains annotated images for 20 object categories.</p>

<h4>2. Kaggle Environment Setup</h4>
<p>I utilized <strong>Kaggle Notebooks</strong> with <strong>GPU acceleration</strong> enabled (Tesla P100 or T4) to speed up model training and reduce computation time.</p>

<h4>3. Cloning the YOLOv5 Repository</h4>
<p>Using a Kaggle code cell, I cloned the official <strong>Ultralytics YOLOv5</strong> repository, which provides state-of-the-art object detection models and tools for training and evaluation.</p>

<h4>4. Training the YOLOv5 Model</h4>
<p>After configuring the dataset in <code>data.yaml</code> and adjusting training parameters (epochs, batch size, etc.), I trained the YOLOv5 model on the Pascal VOC dataset using the GPU runtime.</p>

<h4>5. Model Conversion (.pt to .onnx)</h4>
<p>Once training completed, the best model weights (<code>best.pt</code>) were exported to the <strong>ONNX format</strong> (<code>best.onnx</code>) to make the model compatible with various inference engines and deployable across different platforms.</p>

<h4>6. Downloading Outputs from Kaggle</h4>
<p>I downloaded the <code>runs</code> folder, which contains trained model weights, metrics, and predictions for further evaluation and deployment locally.</p>

<h4>7. Local Predictions with OpenCV</h4>
<p>Back on my local machine, I integrated the <code>best.onnx</code> model into a custom Python script using <strong>OpenCV</strong> for inference. This script supports predictions on images, videos, and real-time webcam input.</p>

<h4>8. Deployment and UI Development</h4>
<p>For a more interactive experience, I began building a <strong>Streamlit interface</strong> and also explored <strong>Kivy</strong> to package the app for Android, enabling users to upload images or use their webcam to perform object detection on-device.</p>
""", unsafe_allow_html=True)

elif st.session_state.section == 'Description':
    st.header("📄 Description")
    with open("summary.html", "r", encoding="utf-8") as file:
        html_content = file.read()

    components.html(html_content, height=500, scrolling=True)
    # Load HTML content
    with open("notebook.html", "r", encoding="utf-8") as file:
        html_content = file.read()

    # Display using iframe-style rendering
    components.html(html_content, height=800, scrolling=True)

elif st.session_state.section == 'Gallery':
    st.header("🖼️ Gallery")
    



    # List of image data with file path and corresponding description
    image_cards = [
    {
        "path": "/home/rgukt/Desktop/objectdetection/App/images/confusion_matrix.png",
        "title": "Confusion Matrix",
        "desc": "This matrix shows how well the model is performing across different object categories."
    },
    {
        "path": "/home/rgukt/Desktop/objectdetection/App/images/F1_curve.png",
        "title": "F1 Curve",
        "desc": "The F1 curve shows the balance between precision and recall for each class across confidence thresholds. A higher F1 score indicates better model performance, especially when class distribution is imbalanced."
    },
    {
        "path": "/home/rgukt/Desktop/objectdetection/App/images/labels_correlogram.jpg",
        "title": "Correlation",
        "desc": "Label correlation highlights how often certain object classes appear together in images. This can help identify class dependencies or co-occurrences, which may affect training and detection accuracy."
    },
    {
        "path": "/home/rgukt/Desktop/objectdetection/App/images/P_curve.png",
        "title": "Precision Curve",
        "desc": "Precision is the ratio of correctly predicted positive observations to total predicted positives. A high precision score means fewer false positives."
    },
    {
        "path": "/home/rgukt/Desktop/objectdetection/App/images/PR_curve.png",
        "title": "Precision vs Recall Curve",
        "desc": "This curve helps visualize the trade-off between precision and recall for various confidence levels. It's crucial for choosing a suitable threshold based on your application needs."
    },
    {
        "path": "/home/rgukt/Desktop/objectdetection/App/images/R_curve.png",
        "title": "Recall Curve",
        "desc": "The recall curve shows how well the model identifies actual objects. A higher recall means fewer missed detections. It’s crucial in applications where detecting every object is important."
    },
    {
        "path": "/home/rgukt/Desktop/objectdetection/App/images/results.png",
        "title": "Results",
        "desc": "The results section summarizes the overall performance of the model across metrics like mAP (mean Average Precision), precision, recall, F1 score, and per-class accuracy."
    },
    {
        "path": "/home/rgukt/Desktop/objectdetection/App/images/train_batch0.jpg",
        "title": "Train Batch 1",
        "desc": "This visual shows sample images from the training batch during training with bounding boxes, helping validate that annotations are correctly applied."
    },
    {
        "path": "/home/rgukt/Desktop/objectdetection/App/images/val_batch0_labels.jpg",
        "title": "Validation Batch 1 Labels ",
        "desc": "Validation batch labels represent the true class annotations for a batch of images used during model validation. They are essential for comparing predicted results against ground truth, helping evaluate the models performance on unseen data."
    },
    {
        "path": "/home/rgukt/Desktop/objectdetection/App/images/val_batch0_pred.jpg",
        "title": "Validation Batch Predictions 1",
        "desc": "Validation batch predictions are the outputs generated by the model for a set of validation images. These predictions are compared against the true labels to assess accuracy, precision, recall, and other performance metrics, offering insights into how well the model generalizes to new, unseen data."
    },
    {
        "path": "/home/rgukt/Desktop/objectdetection/App/images/train_batch1.jpg",
        "title": "Train Batch 1",
        "desc": "This visual shows sample images from the training batch during training with bounding boxes, helping validate that annotations are correctly applied."
    },
    {
        "path": "/home/rgukt/Desktop/objectdetection/App/images/val_batch1_labels.jpg",
        "title": "Validation Batch 1 Labels ",
        "desc": "Validation batch labels represent the true class annotations for a batch of images used during model validation. They are essential for comparing predicted results against ground truth, helping evaluate the models performance on unseen data."
    },
    {
        "path": "/home/rgukt/Desktop/objectdetection/App/images/val_batch1_pred.jpg",
        "title": "Validation Batch Predictions 1",
        "desc": "Validation batch predictions are the outputs generated by the model for a set of validation images. These predictions are compared against the true labels to assess accuracy, precision, recall, and other performance metrics, offering insights into how well the model generalizes to new, unseen data."
    },
    {
        "path": "/home/rgukt/Desktop/objectdetection/App/images/train_batch2.jpg",
        "title": "Train Batch 1",
        "desc": "This visual shows sample images from the training batch during training with bounding boxes, helping validate that annotations are correctly applied."
    },
    {
        "path": "/home/rgukt/Desktop/objectdetection/App/images/val_batch2_labels.jpg",
        "title": "Validation Batch 1 Labels ",
        "desc": "Validation batch labels represent the true class annotations for a batch of images used during model validation. They are essential for comparing predicted results against ground truth, helping evaluate the models performance on unseen data."
    },
    {
        "path": "/home/rgukt/Desktop/objectdetection/App/images/val_batch2_pred.jpg",
        "title": "Validation Batch Predictions 1",
        "desc": "Validation batch predictions are the outputs generated by the model for a set of validation images. These predictions are compared against the true labels to assess accuracy, precision, recall, and other performance metrics, offering insights into how well the model generalizes to new, unseen data."
    },

    
    
    
]

    # Inject CSS only once
    st.markdown("""
        <style>
            .card-container {
                display: flex;
                justify-content: space-between;
                align-items: center;
                border: 2px solid #e0e0e0;
                border-radius: 10px;
                padding: 20px;
                margin: 30px 0;
                box-shadow: 2px 2px 10px rgba(0,0,0,0.1);
                transition: all 0.3s ease-in-out;
            }

            .card-container:hover {
                box-shadow: 4px 4px 15px rgba(0,0,0,0.3);
            }

            .text-content {
                flex: 1;
                padding-right: 20px;
                font-family: sans-serif;
            }

            .image-content {
                overflow: hidden;
                border-radius: 8px;
            }

            .image-content img {
                max-width: 400px;
                border-radius: 8px;
                transition: transform 0.4s ease;
            }

            .image-content img:hover {
                transform: scale(1.1);
            }
        </style>
    """, unsafe_allow_html=True)

    # Loop through each image card and render it
    for card in image_cards:
        if os.path.exists(card["path"]):
            with open(card["path"], "rb") as f:
                data = f.read()
                encoded = base64.b64encode(data).decode()

            st.markdown(f"""
                <div class="card-container">
                    <div class="text-content">
                        <h3>{card["title"]}</h3>
                        <p>{card["desc"]}</p>
                    </div>
                    <div class="image-content">
                        <img src="data:image/png;base64,{encoded}" alt="{card["title"]}">
                    </div>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.error(f"Image not found: {card['path']}")
        

    
    


elif st.session_state.section == 'Model':
    st.header("🤖 Model")

    # Upload Image
    image = st.file_uploader('Upload an image:', type=['png', 'jpg', 'jpeg'])

    if image is not None:
        # Display original image using PIL
        st.image(image, caption="Uploaded Image", width=300)

        # Convert uploaded image to CV2 format
        file_bytes = np.asarray(bytearray(image.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)  # BGR format

        # Object detection (replace this with your YOLO inference)
        img_pred = yolo.predictions(img)  # Assuming your `yolo.predictions()` returns image

        # Convert to RGB for displaying with Streamlit
        image_prediction = cv2.cvtColor(img_pred, cv2.COLOR_BGR2RGB)

        # Display detected image
        st.image(image_prediction, caption="Detection Result", width=300)

    
    uploaded_video = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])

    if uploaded_video is not None:
        # Save to temporary file
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())

        # Open video
        cap = cv2.VideoCapture(tfile.name)
        stframe = st.empty()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Resize frame to smaller size
            resized_frame = cv2.resize(frame, (480, 360))  # Set width x height

            # Run object detection (replace with your actual YOLO model)
            prediction = yolo.predictions(resized_frame)

            # Convert to RGB for display
            frame_rgb = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)

            # Show in Streamlit
            stframe.image(frame_rgb, channels="RGB")

        cap.release()

    
    

    # Custom Video Processor
    class VideoProcessor(VideoProcessorBase):
        def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
            img = frame.to_ndarray(format="bgr24")
            img = cv2.resize(img, (480, 360))

            try:
                detected_frame = yolo.predictions(img)
            except Exception as e:
                print(f"Error in prediction: {e}")
                detected_frame = img

            return av.VideoFrame.from_ndarray(detected_frame, format="bgr24")

    # Streamlit UI
    st.title("📸 Real-time Object Detection")

    # Camera selection
    camera_choice = st.radio("Select Camera", ["Rear Camera", "Front Camera"])

    # Set facingMode based on selection
    facing_mode = "environment" if camera_choice == "Rear Camera" else "user"

    # Launch WebRTC streamer with facingMode setting
    webrtc_streamer(
        key="object-detect",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=VideoProcessor,
        media_stream_constraints={
            "video": {
                "facingMode": facing_mode
            },
            "audio": False,
        },
        async_processing=True,
    )