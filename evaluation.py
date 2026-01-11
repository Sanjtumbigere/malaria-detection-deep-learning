import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# -------------------------
# Paths
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
class_labels = list(test_gen.class_indices.keys())

# -------------------------
# Evaluate each model
# -------------------------
for model_name, model_path in MODELS.items():
    print(f"\n🔹 Evaluating {model_name} model")
    model = load_model(model_path)

    # Predict
    y_pred_prob = model.predict(test_gen)
    y_pred = np.argmax(y_pred_prob, axis=1)
    y_true = test_gen.classes

    # Accuracy
    acc = accuracy_score(y_true, y_pred)
    print(f"✅ Test Accuracy: {acc*100:.2f}%")

    # Classification Report
    report = classification_report(y_true, y_pred, target_names=class_labels)
    print(f"✅ Classification Report for {model_name}:\n", report)
    report_path = os.path.join(RESULTS_DIR, f"classification_report_{model_name}.txt")
    with open(report_path, "w") as f:
        f.write(report)
    print(f"📄 Classification report saved: {report_path}")

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    cm_path = os.path.join(RESULTS_DIR, f"confusion_matrix_{model_name}.png")
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_labels, yticklabels=class_labels)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Confusion Matrix - {model_name}")
    plt.savefig(cm_path)
    plt.close()
    print(f"📊 Confusion matrix saved: {cm_path}")