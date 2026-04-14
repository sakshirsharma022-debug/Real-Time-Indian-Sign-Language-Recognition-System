import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    LSTM, Dense, Dropout, BatchNormalization, Bidirectional
)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.regularizers import l2

# =========================
# CONFIG
# =========================
DATA_PATH = r"C:\Users\Harshit Sharma\OneDrive\Desktop\ISL(V2)\dataset"
SEQUENCE_LENGTH = 30
FEATURE_DIM = 192
EPOCHS = 15
BATCH_SIZE = 16

# =========================
# LOAD DATA
# =========================
labels = sorted(os.listdir(DATA_PATH))
label_map = {label: i for i, label in enumerate(labels)}

X, y = [], []

for label in labels:
    label_path = os.path.join(DATA_PATH, label)
    for seq in os.listdir(label_path):
        seq_path = os.path.join(label_path, seq)

        frames = []
        for f in range(SEQUENCE_LENGTH):
            frame_path = os.path.join(seq_path, f"{f}.npy")
            frames.append(np.load(frame_path))

        X.append(frames)
        y.append(label_map[label])

X = np.array(X, dtype=np.float32)
y = to_categorical(y, num_classes=len(labels))

print("Data Loaded")
print("X shape:", X.shape)
print("y shape:", y.shape)


X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)


model = Sequential([

    Bidirectional(
        LSTM(
            128,
            return_sequences=True,
            activation="tanh",
            kernel_regularizer=l2(1e-4)
        ),
        input_shape=(SEQUENCE_LENGTH, FEATURE_DIM)
    ),
    BatchNormalization(),
    Dropout(0.4),

    Bidirectional(
        LSTM(
            64,
            return_sequences=False,
            kernel_regularizer=l2(1e-4)
        )
    ),
    BatchNormalization(),
    Dropout(0.4),

    Dense(
        64,
        activation="relu",
        kernel_regularizer=l2(1e-4)
    ),
    Dropout(0.3),

    Dense(len(labels), activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=4,
    restore_best_weights=True
)

reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=2,
    min_lr=1e-5
)


history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[early_stop, reduce_lr],
    verbose=1
)


model.save("isl_model_v3_bilstm.h5")
print(" Model saved as isl_model_v3_bilstm.h5")


plt.figure()
plt.plot(history.history["accuracy"], label="Train Acc")
plt.plot(history.history["val_accuracy"], label="Val Acc")
plt.title("Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

plt.figure()
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Val Loss")
plt.title("Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()


y_pred = model.predict(X_val)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_val, axis=1)

cm = confusion_matrix(y_true, y_pred_classes)
disp = ConfusionMatrixDisplay(cm, display_labels=labels)
disp.plot(cmap="Blues", xticks_rotation=45)
plt.title("Confusion Matrix")
plt.show()
