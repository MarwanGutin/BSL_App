from my_classes import *
from utils import *
from model import my_load_model, create_and_train
import mediapipe as mp, numpy as np
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from dataset import load_data
import csv
import os
import numpy as np
from itertools import product


hand_model = mp.solutions.hands.Hands(
    static_image_mode=False, 
    min_detection_confidence=0.5, 
    min_tracking_confidence=0.4
)

pose_model = mp.solutions.pose.Pose(
    static_image_mode=False, 
    min_detection_confidence=0.5, 
    min_tracking_confidence=0.4
)

GIF_FOLDER = GIF_FOLDER = 'sign_gifs/'


if __name__ == '__main__':
    actions = np.array(['A','B','C','D','E','F','G','H','I','J','K','L','M',
                        'N','O','P','Q','R','S','T','U','V','W','X','Y','Z'])
                        
    aug, norm, hands_only, activation = False, True, False, 'relu'
    
    model = my_load_model('best_model.h5')
    app = AppController(actions, model, hand_model, pose_model, GIF_FOLDER, aug, norm, hands_only)
    app.start()