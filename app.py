import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# -------------------------
# Paths & Config
# -------------------------
MODELS = {
    "Custom CNN": r"C:\Users\user\Desktop\finalmalaria\results\cnn_model.h5",
    "MobileNet": r"C:\Users\user\Desktop\finalmalaria\results\mobilenet_model.h5"
}
TEST_DIR = r"C:\Users\user\Desktop\finalmalaria\preprocessed_data\test"
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
RESULTS_DIR = r"C:\Users\user\Desktop\finalmalaria\results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# -------------------------
# Test data generator
# -------------------------
test_datagen = ImageDataGenerator(rescale=1./255)
test_gen = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)
class_names = list(test_gen.class_indices.keys())

# -------------------------
# Streamlit UI
# -------------------------
st.set_page_config(page_title="Malaria Detection", layout="wide")
st.title("🦟 Malaria Detection using Deep Learning")

menu = ["Home", "Evaluate Models", "Upload & Predict", "Reports"]
choice = st.sidebar.selectbox("Menu", menu)

# -------------------------
# Home
# -------------------------
if choice == "Home":
    st.subheader("Welcome")
    st.write("This app detects malaria from microscopic blood smear images using Deep Learning models (Custom CNN & MobileNet).")
    st.image(
        "https://www.cdc.gov/malaria/images/parasites/parasite-blood-smear.jpg",
        caption="Microscopic Malaria Cell",
        use_container_width=True
    )

# -------------------------
# Evaluate Models
# -------------------------
elif choice == "Evaluate Models":
    for model_name, model_path in MODELS.items():
        st.subheader(f"🔹 Evaluating {model_name}")
        model = load_model(model_path)

        # Predict
        y_pred_prob = model.predict(test_gen)
        y_pred = np.argmax(y_pred_prob, axis=1)
        y_true = test_gen.classes  # ✅ FIX: no np.argmax here

        # Accuracy
        acc = accuracy_score(y_true, y_pred)
        st.write(f"✅ Accuracy: *{acc*100:.2f}%*")

        # Classification Report
        report = classification_report(y_true, y_pred, target_names=class_names)
        st.text(report)

        # Confusion Matrix
        cm = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=class_names, yticklabels=class_names, ax=ax)
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title(f"Confusion Matrix - {model_name}")
        st.pyplot(fig, use_container_width=True)

# -------------------------
# Upload & Predict
# -------------------------
elif choice == "Upload & Predict":
    st.subheader("📤 Upload an Image for Prediction")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

    model_choice = st.selectbox("Select Model", list(MODELS.keys()))

    if uploaded_file is not None:
        st.image(uploaded_file, caption="Uploaded Image", use_container_width=250)

        model = load_model(MODELS[model_choice])

        # Preprocess
        img = image.load_img(uploaded_file, target_size=IMG_SIZE)
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Predict
        pred_prob = model.predict(img_array)
        pred_class = np.argmax(pred_prob, axis=1)[0]
        confidence = np.max(pred_prob)

        st.success(f"✅ Prediction: *{class_names[pred_class]}* ({confidence*100:.2f}% confidence)")

# -------------------------
# Reports
# -------------------------
elif choice == "Reports":
    st.subheader("📊 Saved Reports")
    st.write("Classification reports and confusion matrices will be saved in the results folder after evaluation.")
    for model_name in MODELS.keys():
        report_path = os.path.join(RESULTS_DIR, f"classification_report_{model_name}.txt")
        cm_path = os.path.join(RESULTS_DIR, f"confusion_matrix_{model_name}.png")
        if os.path.exists(report_path):
            st.write(f"📄 Report for {model_name}:")
            with open(report_path, "r") as f:
                st.text(f.read())
        if os.path.exists(cm_path):
            st.image(cm_path, caption=f"Confusion Matrix - {model_name}", width=250)
