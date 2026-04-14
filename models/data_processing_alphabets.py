import cv2
import mediapipe as mp
import pandas as pd
import numpy as np
import os
from tqdm import tqdm

# --- CONFIGURATION ---
# Using the absolute path you provided to ensure Windows finds the images
DATA_ROOT = r"C:\Users\Harshit Sharma\OneDrive\Desktop\B.Tech Project\data\raw\asl_alphabet_train\asl_alphabet_train"
LIMIT_PER_CLASS = 300  # Number of images per letter to keep the dataset balanced
# ---------------------

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True, 
    max_num_hands=1, 
    min_detection_confidence=0.7
)

def calculate_agd_features(landmarks):
    """
    RESEARCH CORE: Transforms 21 points into 7 Associated Geometric Descriptors (AGD).
    This is your 'Secret Sauce' for the patent/research paper.
    """
    # Convert landmarks to a 21x3 numpy array
    lms = np.array([[lm.x, lm.y, lm.z] for lm in landmarks])
    
    # 1. Translation Invariance: Move wrist (point 0) to origin (0,0,0)
    lms = lms - lms[0]
    
    # 2. Scale Invariance: Normalize by the length of the palm (wrist to middle finger base)
    hand_size = np.linalg.norm(lms[9]) + 1e-6
    lms = lms / hand_size
    
    # 3. Distance Features: Wrist to each Fingertip
    tips = [4, 8, 12, 16, 20]
    distances = [np.linalg.norm(lms[t]) for t in tips]
    
    # 4. Angular Features: Inter-digital angles (using vector dot products)
    def get_angle(v1_s, v1_e, v2_s, v2_e):
        vec1 = v1_e - v1_s
        vec2 = v2_e - v2_s
        unit1 = vec1 / (np.linalg.norm(vec1) + 1e-6)
        unit2 = vec2 / (np.linalg.norm(vec2) + 1e-6)
        return np.arccos(np.clip(np.dot(unit1, unit2), -1.0, 1.0))

    # Angle between Thumb and Index
    angle_ti = get_angle(lms[2], lms[4], lms[5], lms[8])
    # Angle between Index and Middle
    angle_im = get_angle(lms[5], lms[8], lms[9], lms[12])
    
    return distances + [angle_ti, angle_im]

# Initialize list before the loop to avoid NameError
output_rows = []

print("--- Hand Feature Extraction System ---")

if not os.path.exists(DATA_ROOT):
    print(f"❌ ERROR: Path not found! Looking for: {DATA_ROOT}")
    print("Please check if your dataset is in that exact folder.")
else:
    # Identify subfolders (A, B, C, etc.)
    classes = [d for d in os.listdir(DATA_ROOT) if os.path.isdir(os.path.join(DATA_ROOT, d))]
    print(f"🚀 Found {len(classes)} classes. Starting extraction...")

    for label in tqdm(sorted(classes)):
        label_path = os.path.join(DATA_ROOT, label)
        count = 0
        
        for img_name in os.listdir(label_path):
            if count >= LIMIT_PER_CLASS:
                break
            
            img_path = os.path.join(label_path, img_name)
            img = cv2.imread(img_path)
            if img is None:
                continue
            
            # MediaPipe processing (requires RGB)
            results = hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            
            if results.multi_hand_landmarks:
                landmarks = results.multi_hand_landmarks[0].landmark
                
                # A: Raw Data (63 features: x, y, z for 21 points)
                raw_data = [coord for lm in landmarks for coord in [lm.x, lm.y, lm.z]]
                
                # B: AGD Data (7 features: 5 distances, 2 angles)
                agd_data = calculate_agd_features(landmarks)
                
                output_rows.append([label] + raw_data + agd_data)
                count += 1

    # Save logic
    if len(output_rows) > 0:
        raw_cols = [f'r_{i}_{c}' for i in range(21) for c in ['x','y','z']]
        agd_cols = ['d_thumb', 'd_index', 'd_middle', 'd_ring', 'd_pinky', 'a_ti', 'a_im']
        
        df = pd.DataFrame(output_rows, columns=['label'] + raw_cols + agd_cols)
        
        # Ensure the processed data folder exists
        save_dir = r"C:\Users\Harshit Sharma\OneDrive\Desktop\B.Tech Project\data\processed"
        os.makedirs(save_dir, exist_ok=True)
        
        csv_file = os.path.join(save_dir, "master_features.csv")
        df.to_csv(csv_file, index=False)
        print(f"\n✅ SUCCESS: Created master_features.csv at {csv_file}")
        print(f"📊 Total samples processed: {len(df)}")
    else:
        print("\n❌ FAILED: No data was collected. Ensure images contain clear hands.")