import cv2, time, os, math, random, numpy as np, mediapipe as mp
from utils import *


# collect mp4 videos of user signing
def collect_vids(signs, output_folder, num_videos=20, frames_per_video=20, fps=12):
    """
    Output_folder - destination folder for video files
    """
    resize_dim = (640, 480)
    delay_between_frames = 1/fps
    
    os.makedirs(output_folder, exist_ok=True)
    
    # initiate webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("cannot access webcam!")
        exit()
    quit = False
    
    # countdown before starting collection
    print("Get ready starting in 3 seconds")
    time.sleep(3)
    for sign in signs: 
        if quit:
            break
        os.makedirs(os.path.join(output_folder, sign), exist_ok=True)
        # start at -1 and ignore the first recording
        for video in range(-1, num_videos):
            if quit:
                break
            if video != -1:
                out = cv2.VideoWriter(
                    os.path.join(output_folder, sign, f'{video}.mp4'),
                    cv2.VideoWriter_fourcc(*'mp4v'),
                    fps, # save at 12 fps
                    resize_dim # resize to 640, 480
                )
            
            frame_count = 0
            while frame_count < frames_per_video:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_resized = cv2.resize(frame, resize_dim)
                
                # add 'recording' text to indicate recording
                cv2.putText(frame_resized, "Recording...", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                # only save the frames if its not video -1
                if video != -1:
                    out.write(frame_resized) 
                cv2.imshow('Recording Sign', frame_resized)
                
                # add to frame count
                frame_count += 1
                # sleep to force 12 fps recordings
                time.sleep(delay_between_frames)
                
                # terminate if q is pressed
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    quit = True
                    break
            
            # release the recording 
            if video != -1:
                out.release()
            
            # show 'Video saved'
            saved_message = f"Video {video} saved"
            
            # 1.3 second break between each recording
            start_time = time.time()
            while time.time() - start_time < 1.3:
                ret, frame = cap.read()
                if not ret:
                    break
        
                frame_resized = cv2.resize(frame, resize_dim)
                cv2.putText(frame_resized, saved_message, (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow('Recording Sign', frame_resized)
        
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    quit = True
                    break
        
        pause = True
        while pause:
            if cv2.waitKey(1) & 0xFF == ord(' '):
                pause = False
            elif cv2.waitKey(1) & 0xFF == ord('q'):
                quit = True
                pause = False
                
    cap.release()
    cv2.destroyAllWindows()
    print(f"successfully saved mp4 files at {output_folder}")

def setup_folders(output_path, video_path, actions):
    
    # create a folder for each action, and subfolders for each sequence/video
    for action in actions:
        video_files = sorted_alphanumeric([f for f in os.listdir(os.path.join(video_path, action)) if f.endswith('.mp4')])
        #no_sequences
        for sequence in range(len(video_files)):
            # try-except to catch error in case of preexisting folders
            try:
                os.makedirs(os.path.join(output_path, action, str(sequence)))
            except:
                pass

# use mediapipe to collect keypoints from mp4 files
def collect_keypoints(input_path, output_path, sequence_length, actions):
    """ 
    captures keypoints from mp4 files and saves each frames keypoints as npy
    creates a folder for each sign containing subfolders for each video
    video subfolders contain 20 npy files corresponding to each 20 frames
    """
    
    quit = False
    
    with mp.solutions.hands.Hands(
            static_image_mode=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.4) as hands, \
         mp.solutions.pose.Pose(
            static_image_mode=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.4) as pose:
                for action in actions:
                    if quit:
                        break
                    folder_path = os.path.join(input_path, action)
                    video_files = sorted_alphanumeric([f for f in os.listdir(folder_path) if f.endswith('.mp4')])
                    
                    for sequence, vid in enumerate(video_files):
                        if quit:
                            break
                            
                        cap = cv2.VideoCapture(os.path.join(folder_path, str(vid)))
                        
                        frame_num = 0
                        reset_count = 3
                        lh_count = 3
                        rh_count = 3
                        prev_result = None
                        prev_keypoints = np.zeros(75*3)
                        
                       
                        #generate fixed augmentation params for each video sequence
                        angle = random.uniform(-14, 14)
                        dx = random.randint(-int(0.08*640), int(0.08*640))
                        
                        while cap.isOpened() and frame_num < sequence_length:
                            ret, frame = cap.read()
                            if not ret:
                                break
                            
                            # augment the frame
                            frame = augment_frame(frame, angle, dx)
                                
                            # run mediapipe feature extraction on model
                            hand_results, pose_results = mediapipe_detection(frame, hands, pose)
                            if not hand_results.multi_hand_landmarks:
                                if reset_count < 3:
                                    if prev_result:
                                        hand_results = prev_results
                                        reset_count += 1
                            else:
                                reset_count = 0
                            
                                
                            # show the landmarks
                            draw_styled_landmarks(frame, hand_results, pose_results)
                            
                            # display current recording
                            cv2.putText(frame, f"Collecting {action}, Seq {sequence}, Frame {frame_num}", (15,12),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1, cv2.LINE_AA)
                            cv2.imshow('Video Feed', frame)
                            
                            keypoints = extract_keypoints(hand_results, pose_results)
                            # if left_hand not detected
                            if np.all(keypoints[:63] == 0.0):
                                # if left_hand previously detected
                                if not np.all(prev_keypoints[:63]) and lh_count < 3:
                                    #print("copied left hand")
                                    # copy the previous left hand keypoints
                                    keypoints[:63] = prev_keypoints[:63]
                                    lh_count += 1
                            else:
                                lh_count = 0
                            # if right hand not detected
                            if np.all(keypoints[63:126] == 0.0):
                                # if right hand previously detected
                                if not np.all(prev_keypoints[63:126] == 0.0) and rh_count < 3:
                                    #print("copied right hand")
                                    # copy previous right hand keypoints
                                    keypoints[63:126] = prev_keypoints[63:126]
                                    rh_count += 1
                                else:
                                    rh_count = 0
                            prev_keypoints = keypoints
                            save_path = os.path.join(output_path, action, str(sequence))
                            os.makedirs(save_path, exist_ok=True)
                            np.save(os.path.join(save_path, str(frame_num)), keypoints)
                            
                            frame_num += 1
                            if cv2.waitKey(10) & 0xFF == ord('q'):
                                quit = True
                                break
                        cap.release()
                cv2.destroyAllWindows()

# apply horizontal shift and angle rotation to frames
def augment_frame(frame, angle, dx):
    
    h, w = frame.shape[:2]
    
    # Build the rotation matrix with translation
    centre = (w//2, h//2)
    M = cv2.getRotationMatrix2D(centre, angle, 1.0)
    M[0, 2] += dx # apply horizontal shift to the x translation
    
    # Apply the affine transform
    return cv2.warpAffine(frame, M, (w, h), borderMode=cv2.BORDER_REFLECT)

# read and store the MediaPipe landmark data fron npy files
def load_data(actions, parent_folder="../A_P_MediaPipe", norm=True, hands_only=False):
    sequences = []
    labels = []
    label_map = {label:num for num, label in enumerate(actions)}
    
    # loop through each class
    for action in actions:
        
        folder_path = os.path.join(parent_folder, action)
        sequence_folders = sorted_alphanumeric([f for f in os.listdir(folder_path)])
        
        # loop through each sequence folder
        for num, sequence in enumerate(sequence_folders):
            npy_files = sorted_alphanumeric([f for f in os.listdir(os.path.join(folder_path, str(sequence))) if f.endswith('.npy')])
            window = []
            # loop through each frame .npy file
            for frame_num in range(len(npy_files)):
                res = np.load(os.path.join(folder_path, str(sequence), f'{frame_num}.npy'))
                if hands_only:
                    res = res[:126]

                # apply normalisation
                res = normalise(res, norm, hands_only)
                # add the frame keypoints to current sequence window
                window.append(res)
            
            # add the list of 20 frame keypoints to sequences list
            sequences.append(window)
            labels.append(label_map[action])
            
        
        
        print(f"sign: {action} videos loaded!")
    print(f"keypoints: {len(res)}")
    return sequences, labels

                                
    