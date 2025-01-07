
# Import required libraries
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D, Dropout, Flatten, Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt

# Preprocess dataset for training
def preprocess_images_and_labels(data_path):
    image_data, labels = [], []
    for category in ["with_mask", "without_mask"]:
        path = os.path.join(data_path, category)
        for image_name in os.listdir(path):
            img_path = os.path.join(path, image_name)
            image = cv2.imread(img_path)
            image = cv2.resize(image, (224, 224))
            image_data.append(image)
            labels.append(category)

    return (np.array(image_data), np.array(labels))

# Build the mask detection model
def build_mask_detection_model():
    base_model = MobileNetV2(weights="imagenet", include_top=False, input_tensor=Input(shape=(224, 224, 3)))
    head_model = base_model.output
    head_model = AveragePooling2D(pool_size=(7, 7))(head_model)
    head_model = Flatten()(head_model)
    head_model = Dense(128, activation="relu")(head_model)
    head_model = Dropout(0.5)(head_model)
    head_model = Dense(2, activation="softmax")(head_model)

    return Model(inputs=base_model.input, outputs=head_model)

# Train and save the model
def train_and_save_model(data_path, model_output_path):
    print("[INFO] Loading dataset...")
    data, labels = preprocess_images_and_labels(data_path)

    data = data / 255.0
    label_binarizer = LabelBinarizer()
    labels = label_binarizer.fit_transform(labels)
    labels = np.array(labels)

    train_x, test_x, train_y, test_y = train_test_split(data, labels, test_size=0.2, stratify=labels, random_state=42)

    print("[INFO] Compiling model...")
    model = build_mask_detection_model()
    optimizer = Adam(learning_rate=0.001)
    model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"])

    print("[INFO] Training model...")
    history = model.fit(train_x, train_y, batch_size=32, epochs=20, validation_data=(test_x, test_y))

    print("[INFO] Evaluating model...")
    predictions = model.predict(test_x, batch_size=32)
    print(classification_report(test_y.argmax(axis=1), predictions.argmax(axis=1), target_names=label_binarizer.classes_))

    print("[INFO] Saving model...")
    model.save(model_output_path)
