import numpy as np

def get_normalized_rays(hand_lms):
    # Wrist is the origin (0,0)
    wrist = hand_lms.landmark[0]
    
    
    middle_base = hand_lms.landmark[9]
    hand_size = np.sqrt((middle_base.x - wrist.x)**2 + (middle_base.y - wrist.y)**2) + 1e-6
    
    
    rays = []
    for lm in hand_lms.landmark:
        dist = np.sqrt((lm.x - wrist.x)**2 + (lm.y - wrist.y)**2)
        rays.append(dist / hand_size)
    return rays