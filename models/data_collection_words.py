import cv2
import mediapipe as mp
import numpy as np
import os
import time

# =========================
# CONFIG
# =========================
TARGET_WORD = "HELP"
REAL_TAKES = 5
AUGMENT_FACTOR = 5
SEQUENCE_LENGTH = 30
FEATURE_DIM = 192

DATA_PATH = r"C:\Users\Harshit Sharma\OneDrive\Desktop\ISL(V2)\dataset"
os.makedirs(os.path.join(DATA_PATH, TARGET_WORD), exist_ok=True)

# =========================
# MEDIAPIPE SETUP
# =========================
mp_hands = mp.solutions.hands
mp_face = mp.solutions.face_mesh
mp_pose = mp.solutions.pose

hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

face = mp_face.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

pose = mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

cap = cv2.VideoCapture(0)
recorded_data = []

# =========================
# LANDMARK SELECTION
# =========================
FACE_IDX = [1, 33, 61, 199, 263, 291, 13, 14, 0, 17]
POSE_IDX = [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27]

print(f"🎥 Recording: {TARGET_WORD}")

# =========================
# RECORDING LOOP
# =========================
for take in range(1, REAL_TAKES + 1):
    time.sleep(1)

    for i in range(3, 0, -1):
        ret, frame = cap.read()
        cv2.putText(frame, f"TAKE {take}/{REAL_TAKES}", (40, 40), 1, 2, (255,255,0), 2)
        cv2.putText(frame, f"Ready {i}", (220, 250), 1, 5, (0,0,255), 5)
        cv2.imshow("Recorder", frame)
        cv2.waitKey(700)

    sequence = []
    missing = 0

    for _ in range(SEQUENCE_LENGTH):
        ret, frame = cap.read()
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        res_hands = hands.process(image)
        res_face = face.process(image)
        res_pose = pose.process(image)

        frame_features = []

        # =========================
        # HANDS (LEFT + RIGHT)
        # =========================
        hands_data = np.zeros((2, 21, 3))

        if res_hands.multi_hand_landmarks:
            for idx, hand_landmarks in enumerate(res_hands.multi_hand_landmarks[:2]):
                wrist = hand_landmarks.landmark[0]
                for i, lm in enumerate(hand_landmarks.landmark):
                    hands_data[idx, i] = [
                        lm.x - wrist.x,
                        lm.y - wrist.y,
                        lm.z - wrist.z
                    ]

        frame_features.extend(hands_data.flatten())

        # =========================
        # FACE (SELECTED POINTS)
        # =========================
        face_data = np.zeros((len(FACE_IDX), 3))
        if res_face.multi_face_landmarks:
            face_landmarks = res_face.multi_face_landmarks[0]
            nose = face_landmarks.landmark[1]
            for i, idx in enumerate(FACE_IDX):
                lm = face_landmarks.landmark[idx]
                face_data[i] = [
                    lm.x - nose.x,
                    lm.y - nose.y,
                    lm.z - nose.z
                ]
        frame_features.extend(face_data.flatten())

        # =========================
        # POSE (UPPER BODY)
        # =========================
        pose_data = np.zeros((len(POSE_IDX), 3))
        if res_pose.pose_landmarks:
            hip = res_pose.pose_landmarks.landmark[23]
            for i, idx in enumerate(POSE_IDX):
                lm = res_pose.pose_landmarks.landmark[idx]
                pose_data[i] = [
                    lm.x - hip.x,
                    lm.y - hip.y,
                    lm.z - hip.z
                ]
        frame_features.extend(pose_data.flatten())

        frame_features = np.array(frame_features)

        if frame_features.shape[0] != FEATURE_DIM:
            missing += 1
            frame_features = np.zeros(FEATURE_DIM)

        sequence.append(frame_features)

        cv2.rectangle(frame, (0,0), (640,40), (0,255,0), -1)
        cv2.putText(frame, "RECORDING", (200,30), 1, 2, (0,0,0), 2)
        cv2.imshow("Recorder", frame)
        cv2.waitKey(1)

    if missing > 6:
        print(f" Skipped TAKE {take} (poor detection)")
        continue

    recorded_data.append(np.array(sequence))

cap.release()
cv2.destroyAllWindows()

# =========================
# SAVE + AUGMENT
# =========================
print(" Saving sequences...")
idx = 0

for seq in recorded_data:
    base_path = os.path.join(DATA_PATH, TARGET_WORD, str(idx))
    os.makedirs(base_path, exist_ok=True)

    for f, data in enumerate(seq):
        np.save(os.path.join(base_path, f"{f}.npy"), data)
    idx += 1

    for _ in range(AUGMENT_FACTOR):
        jittered = seq + np.random.normal(0, 0.002, seq.shape)
        aug_path = os.path.join(DATA_PATH, TARGET_WORD, str(idx))
        os.makedirs(aug_path, exist_ok=True)

        for f, data in enumerate(jittered):
            np.save(os.path.join(aug_path, f"{f}.npy"), data)
        idx += 1

print(f" Completed collection for '{TARGET_WORD}'")
