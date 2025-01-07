
# Import required libraries
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import imutils
import time
import cv2
import os

# Function to analyze the input frame and detect mask presence
def analyze_frame_for_masks(frame_data, face_model, mask_model):
    # Get frame dimensions and create a blob for processing
    (height, width) = frame_data.shape[:2]
    blob_data = cv2.dnn.blobFromImage(frame_data, 1.0, (300, 300), (104.0, 177.0, 123.0))
    face_model.setInput(blob_data)
    face_predictions = face_model.forward()

    face_boxes = []
    face_images = []
    mask_predictions = []

    for i in range(0, face_predictions.shape[2]):
        confidence = face_predictions[0, 0, i, 2]
        if confidence > 0.5:  # Confidence threshold
            box = face_predictions[0, 0, i, 3:7] * np.array([width, height, width, height])
            (start_x, start_y, end_x, end_y) = box.astype("int")

            if start_x < 0 or start_y < 0 or end_x > width or end_y > height:
                continue

            face = frame_data[start_y:end_y, start_x:end_x]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)

            face_boxes.append((start_x, start_y, end_x, end_y))
            face_images.append(face)

    if len(face_images) > 0:
        face_images = np.array(face_images, dtype="float32")
        mask_predictions = mask_model.predict(face_images, batch_size=32)

    return (face_boxes, mask_predictions)

# Stream video and detect masks in real-time
def run_video_mask_detection(face_model_path, mask_model_path):
    print("[INFO] Loading models...")
    face_detector = cv2.dnn.readNet(face_model_path)
    mask_detector = load_model(mask_model_path)

    print("[INFO] Starting video stream...")
    video_stream = VideoStream(src=0).start()
    time.sleep(2.0)

    while True:
        frame = video_stream.read()
        frame = imutils.resize(frame, width=800)
        face_boxes, mask_predictions = analyze_frame_for_masks(frame, face_detector, mask_detector)

        for (box, pred) in zip(face_boxes, mask_predictions):
            (start_x, start_y, end_x, end_y) = box
            (mask, without_mask) = pred

            label = "Mask" if mask > without_mask else "No Mask"
            color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

            cv2.putText(frame, label, (start_x, start_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
            cv2.rectangle(frame, (start_x, start_y), (end_x, end_y), color, 2)

        cv2.imshow("Mask Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()
    video_stream.stop()
