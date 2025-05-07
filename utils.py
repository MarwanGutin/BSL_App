import cv2, os, time, re, numpy as np, mediapipe as mp
#opencv, os - file management, time - sleep between frames, 
#numpy - arrays datasets, mediapipe

mp_holistic = mp.solutions.holistic # holistic model - to make detections
mp_drawing = mp.solutions.drawing_utils # drawing utilities - to show detections


# sort a given list in ascending alphanumeric order
def sorted_alphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(data, key=alphanum_key)

# run mediapipe detection to extract keypoints
def mediapipe_detection(image, hand_model, pose_model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # convert image BGR->RGB
    image.flags.writeable = False # make it unwritable - save space
    hand_results = hand_model.process(image) # make hand detections/predictions
    pose_results = pose_model.process(image) # make pose detections/predictions
    image.flags.writeable = True # make it writable again
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # convert back to BGR
    return hand_results, pose_results # return results

# used during data gathering to ensure keypoints are collected
def draw_styled_landmarks(image, hand_results, pose_results):
    # utilises mediapipes drawing solution to display keypoints and connections
    if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=1, circle_radius=1),
                    mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=1, circle_radius=1)
                )
    if pose_results.pose_landmarks:
        mp_drawing.draw_landmarks(
                image, pose_results.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=1, circle_radius=1),
                mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=1, circle_radius=1)
            )

# extracting hand and pose keypoints and concatenating them                  
def extract_keypoints(hand_results, pose_results, norm=False, hands_only=False):
    """ Parameters:
        hand_results    - landmarks gathered from MediaPipe Hands
        pose_results    - landmarks gathered from MediaPipe Hands
        norm            - Boolean True if system is using normalised keypoints else False
        hands_only      - Boolean True if using only hand keypoints False if also using body keypoints
    """
    # initialise empty hands incase they're not detected
    lh = np.zeros(21*3)
    rh = np.zeros(21*3)
    
    # check for hand keypoints
    if hand_results.multi_hand_landmarks and hand_results.multi_handedness:
        for idx, hand_handedness in enumerate(hand_results.multi_handedness):
            hand_label = hand_handedness.classification[0].label # 'Left' or 'Right'
            hand_landmarks = hand_results.multi_hand_landmarks[idx]
            # if detected update the hand keypoints list (lh/rh)
            if hand_label == 'Left':
                lh = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]).flatten()
            elif hand_label == 'Right':
                rh = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]).flatten()
    # if the system is also using body keypoints
    if True:
        # initialise empty pose list
        pose = np.zeros(33*3)
        # check for pose keypoints detected
        if pose_results.pose_landmarks:
            # update the pose keypoints list
            pose = np.array([[lm.x, lm.y, lm.z] for lm in pose_results.pose_landmarks.landmark]).flatten()
        # normalise the keypoints before returning
        keypoints = normalise(np.concatenate([lh, rh, pose]), norm, hands_only)
    else:
        keypoints = normalise(np.concatenate([lh, rh]), norm, hands_only)
    return keypoints

# normalisation of keypoints
def normalise(results, norm, hands_only):
    """ Parameters:
        results     - Extracted and concatenated keypoints from extract_keypoints()
        norm        - Boolean True if system is using normalised keypoints else False
        hands_only  - Boolean True if using only hand keypoints False if also using body keypoints
    """
    if not norm:
        return results
    else:
        keypoints = np.array(results)
        
        # apply body centre normalisation first if pose keypoints included in input
        if not hands_only:
            pose = np.array(keypoints[126:]) # extract only the pose
            if pose.shape[0] != 99:
                raise ValueError("Expected 33 pose landmark (99 values), got {}".format(pose.shape[0]))
                
            left_shoulder = np.array(pose[33:36])
            right_shoulder = np.array(pose[36:39])
                
            head_centre = (pose[6:9] + pose[15:18]) / 2
            body_centre = (left_shoulder + right_shoulder) / 2
            scale = np.linalg.norm(head_centre-body_centre) + 1e-6
                
            # normalise all landmarks to centre
            for i in range(0, len(keypoints), 3):
                keypoints[i:i+3] = (keypoints[i:i+3] - body_centre) / scale
                
            
        #elif norm == 'wrist':
           # normalise hand landmarks to wrists
           # extract wrist positions (landmark 0)
           
        # always apply wrist relative normalisation
        if True:
            lh_wrist = keypoints[0:3]
            rh_wrist = keypoints[63:66]
                    
            # normalise left hand (landmarks 0-62)
            for i in range (0, 63, 3):
                keypoints[i:i+3] -= lh_wrist
                    
            # normalise right hand (landmarks 0-62)
            for i in range (63, 126, 3):
                keypoints[i:i+3] -= rh_wrist
            
    return keypoints

# Draws a probability visualisation of top 3 predicted classes
def prob_viz(res, actions, input_frame):
    """ 
    Parameters:
    res: output from model inference
    actions: array of classes
    input_frame: current image displayed by CV2
    """
    colours = [(117,245,16), (245,180,16), (245,117,16)]
    output_frame = input_frame.copy()
    # Get indices of top 5 probabilities
    top_indices = np.argsort(res)[-3:][::-1]  # descending order

    # Get top 3 probabilities and their indices
    top_probs = res[top_indices]
    
    # display top 3
    for num, prob in enumerate(zip(top_indices, top_probs)):
        gesture = prob[0]
        prob = prob[1]
        cv2.rectangle(output_frame, (0,60+num*40), (int(prob*100), 90+num*40), colours[num], -1)
        cv2.putText(output_frame, actions[gesture], (0, 85+num*40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
        
    return output_frame