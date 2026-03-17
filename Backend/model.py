import os
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Bidirectional
from tensorflow.keras.utils import to_categorical

DATA_PATH = "dataset"

SEQUENCE_LENGTH = 30
FEATURE_DIM = 384
EPOCHS = 20
BATCH_SIZE = 16

labels = sorted(os.listdir(DATA_PATH))
label_map = {label: i for i, label in enumerate(labels)}

X = []
y = []

# ---------------------------
# LOAD DATA
# ---------------------------

for label in labels:

    label_path = os.path.join(DATA_PATH, label)

    for seq in os.listdir(label_path):

        seq_path = os.path.join(label_path, seq)

        frames = []

        for f in range(SEQUENCE_LENGTH):

            frame = np.load(os.path.join(seq_path, f"{f}.npy"))
            frames.append(frame)

        X.append(frames)
        y.append(label_map[label])

X = np.array(X, dtype=np.float32)
y = to_categorical(y, len(labels))

print("X shape:", X.shape)

# ---------------------------
# TRAIN TEST SPLIT
# ---------------------------

X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=np.argmax(y, axis=1),
    random_state=42
)

# ---------------------------
# MODEL
# ---------------------------

model = Sequential([

    Bidirectional(LSTM(128, return_sequences=True),
                  input_shape=(SEQUENCE_LENGTH, FEATURE_DIM)),

    BatchNormalization(),
    Dropout(0.4),

    Bidirectional(LSTM(64)),

    BatchNormalization(),
    Dropout(0.4),

    Dense(64, activation="relu"),
    Dropout(0.3),

    Dense(len(labels), activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ---------------------------
# TRAIN
# ---------------------------

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE
)

# ---------------------------
# SAVE MODEL
# ---------------------------

model.save("isl_model_v4_motion.h5")
print("Model saved")

# ---------------------------
# PREDICTIONS
# ---------------------------

y_pred_prob = model.predict(X_val)
y_pred = np.argmax(y_pred_prob, axis=1)
y_true = np.argmax(y_val, axis=1)

# ---------------------------
# METRICS
# ---------------------------

accuracy = accuracy_score(y_true, y_pred)

print("\nFinal Accuracy:", accuracy)

print("\nClassification Report:\n")
print(classification_report(y_true, y_pred, target_names=labels))

print("\nConfusion Matrix:\n")
print(confusion_matrix(y_true, y_pred))

# ---------------------------
# PLOTS
# ---------------------------

plt.figure()
plt.plot(history.history["accuracy"], label="train_accuracy")
plt.plot(history.history["val_accuracy"], label="val_accuracy")
plt.legend()
plt.title("Model Accuracy")
plt.show()

plt.figure()
plt.plot(history.history["loss"], label="train_loss")
plt.plot(history.history["val_loss"], label="val_loss")
plt.legend()
plt.title("Model Loss")
plt.show()
