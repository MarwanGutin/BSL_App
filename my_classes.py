import cv2, random, time, numpy as np, tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
from utils import *


class AppController:
    def __init__(self, actions, model, hand_model, pose_model, gif_folder, aug, norm, hands_only):
        # save important attributes
        self.actions = actions
        self.model = model
        self.hand_model = hand_model
        self.pose_model = pose_model
        self.gif_folder = gif_folder
        
        self.aug = aug
        self.norm = norm
        self.hands_only = hands_only
        
        self.window = None
    
    def start(self):
        # create the main menu window
        self.window = tk.Tk()
        self.window.title("Welcome to BSL Learning App")
        self.window.geometry('800x600')
        
        # title label
        title_label = ttk.Label(self.window, text="Choose Mode", font=('Helvetica', 24))
        title_label.pack(pady=50)
        
        # learning mode button
        learning_button = ttk.Button(self.window, text="Learning Mode", command=self.start_learning)
        learning_button.pack(pady=20)
        
        # Testing mode button
        testing_button = ttk.Button(self.window, text="Testing Mode", command=self.start_testing)
        testing_button.pack(pady=20)
        
        self.window.mainloop()
     
    def start_learning(self):
        # close main menu and launch learning mode
        self.window.destroy()
        learning_mode = LearningMode(self.gif_folder, self.start)
        learning_mode.start()
    
    def start_testing(self):
        # close main menu and launch testing mode
        self.window.destroy()
        testing_mode = TestingMode(self.actions, self.model, self.hand_model, self.pose_model, self.start, self.aug, self.norm, self.hands_only)
        testing_mode.start()

class TestingMode:
    def __init__(self, actions, model, hand_model, pose_model, main_menu_callback, aug, norm, hands_only):
        self.actions = actions
        self.model = model
        #self.holistic_model = holistic_model
        self.hand_model = hand_model
        self.pose_model = pose_model
        self.main_menu_callback = main_menu_callback
        
        self.aug = aug
        self.norm = norm
        self.hands_only = hands_only
        
        self.window = None
        self.cap = None
        self.running = False
        
        self.sequence = []
        self.sentence = []
        self.predictions = []
        self.reset = 3
        self.lh_count = 3
        self.rh_count = 3
        self.in_frame = False
        # instantiate res to avoid errors with utils.prob_viz before predictions are made
        self.res = np.array([0*len(self.actions)])
        self.prev_keypoints = np.zeros(75*3)
        
        self.score = 0
        self.target_sign = None
        self.round_active = False
        self.result_message = ["", (255,255,255)]
        self.dims = (640,480)
              
    def start(self):
        # create a new window
        self.window = tk.Tk()
        self.window.title("BSL Signing Game")
        self.window.geometry("800x600")
        
        # add a title
        title_label = ttk.Label(self.window, text="Welcome to the BSL Signing Game", font=('Helvetica', 24))
        title_label.pack(pady=20)
        
        # add camera display label
        self.video_label = ttk.Label(self.window)
        self.video_label.pack()
        
        # add return button
        button_frame = ttk.Frame(self.window)
        button_frame.pack(pady=20)
        
        return_button = ttk.Button(button_frame, text="Return to Menu", command=self.stop)
        return_button.pack()
        
        # open webcam
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FPS, 12)
        
        # start the game loop
        self.running = True
        self.new_round()
        self.update_frame()
        
        # handle window closing
        self.window.protocol('WM_DELETE_WINDOW', self.stop)
        self.window.mainloop()
        
    def update_frame(self):
        if self.running and self.cap is not None:
            start_time = time.time() # used for FPS
            ret, frame = self.cap.read()
            if ret:
                
                frame = cv2.resize(frame, (640, 480))
                # run mediapipe detection
                hand_results, pose_results, = mediapipe_detection(frame, self.hand_model, self.pose_model)
                
                # check if hands are detected
                if hand_results.multi_hand_landmarks:
                    # if they werent previously detected, user has reset and we can start detecting again
                    if not self.in_frame:
                        self.result_message = ["", (255,255,255)]
                        self.reset = 0
                    self.in_frame = True
                # if hands not detected
                else:
                    if self.reset < 3:
                        self.reset += 1
                        if len(self.sequence) > 2:
                            self.sequence.append(self.sequence[-1])
                    else:
                        # delete current sequence
                        self.in_frame = False
                        self.sequence = [] 
                
                # record keypoints
                if self.in_frame and self.reset < 3:
                    keypoints = extract_keypoints(hand_results, pose_results, self.norm, self.hands_only)
                    # --- occlusion handling
                    # if left hand not detected
                    if np.all(keypoints[:63] == 0.0):
                        # if left hand previously detected
                        if not np.all(self.prev_keypoints[:42]) and self.lh_count < 3:
                            # copy the previous left hand keypoints
                            keypoints[:63] = self.prev_keypoints[:63]
                            self.lh_count += 1
                        else:
                            self.lh_count = 0
                    # if right hand not detected
                    if np.all(keypoints[63:126] == 0.0):
                        # if right hand previously detected
                        if not np.all(self.prev_keypoints[63:126] == 0.0) and self.rh_count < 3:
                            # copy previous right hand keypoints
                            keypoints[63:126] = self.prev_keypoints[42:84]
                            self.rh_count += 1
                        else:
                            self.rh_count = 0
                            
                    prev_keypoints = keypoints

                    self.sequence.append(keypoints)
                
                # make a prediction if enough frames recorded
                if len(self.sequence) >= 20 and self.reset < 3:
                    # user must reset their hands to record the next sign after this
                    self.reset = 3
 
                    # ensure sequence is only 20 frames
                    self.sequence = np.array(self.sequence[-20:])
                    self.res = self.model.predict(np.expand_dims(self.sequence, axis=0))[0]
                    self.predictions.append(np.argmax(self.res))
                    
                    # model inference to get predicted sign
                    predicted_sign = self.actions[np.argmax(self.res)]
                    
                    # evaluate the result
                    if predicted_sign == self.target_sign and self.round_active:
                        print("Correct Sign!")
                        self.score += 1
                        self.round_active = False
                        self.result_message = ["Correct!", (20,255,20)]
                        self.window.after(1000, self.new_round)
                    else:
                        print("Wrong sign!")
                        self.result_message = ["Try again!", (20,20,255)]
                    
                    # add new prediction
                    self.sentence.append(predicted_sign)
                    # max sentence length = 5
                    if len(self.sentence) > 5: self.sentence = self.sentence[-5:]
                    # reset sequence 
                    self.sequence = []
                
                # draw text overlays
                cv2.putText(frame, f"Target: {self.target_sign}", (10,self.dims[1]-20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2, cv2.LINE_AA)
                cv2.putText(frame, f"Score: {self.score}", (self.dims[0]-140,self.dims[1]-20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2, cv2.LINE_AA) 
                cv2.putText(frame, self.result_message[0], (int(self.dims[0]/2)-60, self.dims[1]-20),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, self.result_message[1], 3, cv2.LINE_AA)  
                
                # show probabilities of top 3 predictions
                image = prob_viz(self.res, self.actions, frame)
                
                # show sequence number to indicate recording of sign
                sequence_text = f"sequence {len(self.sequence)}"
                cv2.putText(image, sequence_text, (self.dims[0]-160, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 2, cv2.LINE_AA)
                
                # calculate and show FPS
                end_time = time.time()
                fps = 1 / (end_time - start_time + 1e-6)
                fps_text = f"FPS: {fps:.2f}"
                
                cv2.putText(image, fps_text, (self.dims[0]-160, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2, cv2.LINE_AA)
               
                # convert to Tkinter image and display
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(image)
                imgtk = ImageTk.PhotoImage(image=img)
                self.video_label.imgtk = imgtk
                self.video_label.configure(image=imgtk)
                
            
            # schedule the next frame
            self.video_label.after(60, self.update_frame)
                    
    def new_round(self):
        # pick a new random sign
        self.target_sign = random.choice(self.actions)
        self.round_active = True
        print(f"New target: {self.target_sign}")
        self.result_message = ["", (255,255,255)]
    
    def stop(self):
        # clean up and return to menu
        self.running = False
        self.round_active = False
        if self.cap:
            self.cap.release()
        if self.window:
            self.window.destroy()
        self.main_menu_callback()
        
class LearningMode:
    def __init__(self, gif_folder, main_menu_callback):
        # initialise important attributes
        self.gif_folder = gif_folder
        self.main_menu_callback = main_menu_callback
        
        self.window = None
        self.cap = None
        self.running = False
        
        self.frames = []
        self.frame_count = []
        self.index = []
        self.gif_label = None
        self.webcam_label = None
        self.current_sign_label = None
        self.flip = False
        self.target_size = (558, 314)
        
    def start(self):
        # create learning window
        self.window = tk.Tk()
        self.window.title("Learning mode")
        self.window.geometry('1200x700')
        
        # layout setup
        top_frame = tk.Frame(self.window)
        top_frame.pack(pady=10)
        
        middle_frame = tk.Frame(self.window)
        middle_frame.pack(pady=10)
        
        bottom_frame = tk.Frame(self.window)
        bottom_frame.pack(pady=10)
        
        
        # --- top frame: buttons for each letter/sign
        letters = ['A','B','C','D','E','F','G','H','I',
                   'J','K','L','M','N','O','P','Q','R',
                   'S','T','U','V','W','X','Y','Z']
                   
        first_row = tk.Frame(top_frame)
        first_row.pack()
        
        second_row = tk.Frame(top_frame)
        second_row.pack()
        
        third_row = tk.Frame(top_frame)
        third_row.pack()
        
        for letter in letters[:9]:
            button = ttk.Button(first_row, text=letter, command=lambda l=letter: self.change_gif(l))
            button.pack(side='left', padx=5, pady=2)
        for letter in letters[9:18]:
            button = ttk.Button(second_row, text=letter, command=lambda l=letter: self.change_gif(l))
            button.pack(side='left', padx=5, pady=2)
        for letter in letters[18:]:
            button = ttk.Button(third_row, text=letter, command=lambda l=letter: self.change_gif(l))
            button.pack(side='left', padx=5, pady=2)
        
        # --- middle-left frame: GIF + label
        left_frame = tk.Frame(middle_frame)
        left_frame.pack(side='left', padx=5)
        
        self.current_sign_label = ttk.Label(left_frame, text="Now learning: A", font=('Helvetica', 20))
        self.current_sign_label.pack(pady=10)
        
        self.gif_label = ttk.Label(left_frame)
        self.gif_label.pack()
        
        # --- middle-right frame: webcam
        right_frame = tk.Frame(middle_frame)
        right_frame.pack(side='right', padx=20)
        
        self.webcam_label = ttk.Label(right_frame)
        self.webcam_label.pack()
        
        # --- bottom frame: flip & back button
        flip_button = ttk.Button(bottom_frame, text="Flip images", command=self.toggle_flip)
        flip_button.pack(side='left', pady=10)
        
        back_button = ttk.Button(bottom_frame, text="Back to Menu", command=self.stop)
        back_button.pack(side='right', pady=10)
        
        # open the webcam
        self.cap = cv2.VideoCapture(0)
        
        # load default gif 
        self.load_gif(os.path.join(self.gif_folder, 'A.gif'))
        
        # start frame updates
        self.running = True
        self.update_frame()
        
        self.window.protocol('WM_DELETE_WINDOW', self.stop)
        self.window.mainloop()
    
    def update_frame(self):
        if self.running:
            # update gif frame
            if self.gif_frames:
                pil_img = self.gif_frames[self.gif_index]
                if self.flip: pil_img = pil_img.transpose(Image.FLIP_LEFT_RIGHT)
                gif_imgtk = ImageTk.PhotoImage(pil_img)
                
                self.gif_label.imgtk = gif_imgtk
                self.gif_label.configure(image=gif_imgtk)
                # keep looping the gif
                self.gif_index = (self.gif_index + 1) % self.gif_frame_count
            
            # update webcam frame
            ret, frame = self.cap.read()
            if ret:
                if self.flip: frame = cv2.flip(frame, 1)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame)
                imgtk = ImageTk.PhotoImage(image=img)
                self.webcam_label.imgtk = imgtk
                self.webcam_label.configure(image=imgtk)
                
            # schedule next update
            self.window.after(30, self.update_frame)
            
    def load_gif(self, path):
        # load and resize all frames of the gif
        gif = Image.open(path)
        self.gif_frames = []
        
        try:
            while True:
                gif_frame = gif.copy()
                # resize while maintining quality with LANCZOS
                gif_frame = gif_frame.resize(self.target_size, Image.Resampling.LANCZOS)
                #self.gif_frames.append(ImageTk.PhotoImage(gif_frame))
                self.gif_frames.append(gif_frame)
                # load the next frame
                gif.seek(len(self.gif_frames))
        except EOFError:
            pass
        
        self.gif_frame_count = len(self.gif_frames)
        self.gif_index = 0
    
    def change_gif(self, sign):
        # change displayed gif when letter button is clicked
        gif_path = os.path.join(self.gif_folder, f'{sign}.gif')
        self.load_gif(gif_path)
        self.current_sign_label.config(text=f"Now Learning: {sign}")
    
    def toggle_flip(self):
        self.flip = not self.flip
    
    def stop(self):
        # stop learning mode and return to menu
        self.running = False
        if self.cap:
            self.cap.release()
        if self.window:
            self.window.destroy()
        self.main_menu_callback()
             