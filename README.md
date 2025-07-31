# BSL Recognition App

This is a British Sign Language (BSL) recognition app that uses MediaPipe to extract hand and pose landmarks, processes these keypoints, and classifies signs using a trained LSTM-based deep learning model.

## Features

- Real-time webcam-based sign recognition (A–Z signs)
- Uses MediaPipe Hands and Pose for landmark detection
- Supports both hand-only and full-body landmark modes
- Model trained with augmented and normalized landmark data
- Optional visualization of prediction probabilities
- Custom LSTM, GRU, and Bidirectional architectures

---

## Project Structure
```text
─── marwangutin-bsl_app/
    ├── best_model.h5
    ├── dataset.py
    ├── LICENSE
    ├── main.py
    ├── model.py
    ├── mp_files.zip
    ├── my_classes.py
    └── utils.py
```

---

## Getting Started

### 1. Clone the Repo

```bash
git clone https://github.com/MarwanGutin/BSL_App.git
cd BSL_App
```

### 2. Install Requirements
Ensure you have Python 3.8+ installed.

```bash
pip install -r requirements.txt
```
If requirements.txt is missing, install manually:

```bash
pip install mediapipe opencv-python tensorflow numpy matplotlib
```
### 3. Run the App
```bash
python main.py
```
The app will activate your webcam and start real-time sign recognition.

## Usage Tips
For the best recognition accuracy:

Ensure your full upper body and face are clearly visible in the webcam frame.

Avoid cluttered backgrounds or poor lighting that may confuse landmark detection.

Stand or sit around 1–1.5 meters from the camera.

Keep your hands, shoulders, and face unobstructed at all times.

Sign at a steady pace — avoid overly fast or abrupt movements.

**The app relies on MediaPipe’s hand and pose models, to predict signs. Proper visibility is key!**

## How It Works
The system recognizes signs by capturing 20 frames of hand and pose landmarks during each signing attempt. Here's how the process works:

Detection begins automatically when your hands enter the frame.

The system collects exactly 20 consecutive frames while your hands remain visible.

Once 20 valid frames are captured, the app uses them to predict the sign.

If your hands leave the frame before 20 frames are collected, the partial sequence is discarded, and the system resets until your hands re-enter.

This approach ensures each prediction is based on a complete, clean gesture with minimal noise.

## Training a New Model
To train a new model on your own dataset:

Place your landmark .npy sequences under a subdirectory in mp_files/ (one folder per class).

Edit model.py to change model architecture or parameters.

Run training:

```bash
python model.py
```
Trained weights will be saved in _models/.

## Utility Scripts
utils.py: Keypoint extraction, normalization, landmark drawing, probability visualization.

dataset.py: Data loading and preprocessing.

model.py: Model building, training, and evaluation utilities.

## Visualization
Top 3 predicted signs are visualized with confidence bars on each frame:
```text
🟩 HELLO   |███████████████████
🟧 YES     |██████
🟥 THANKS  |█
```
## Model Architecture
Default model is a stacked LSTM network with dropout and dense layers. Alternative architectures (GRU, Bidirectional LSTM) are also available via:

```python
create_model(model_type='gru')  # or 'bilstm'
```
## Data Format
Input: sequences of 20 frames, each containing 147 landmark values (hands + pose).

Output: one-hot encoded vector of class labels.

## Sample Workflow
```python
# Run detection
hand_results, pose_results = mediapipe_detection(image, hands, pose)

# Extract and normalize keypoints
keypoints = extract_keypoints(hand_results, pose_results, norm=True)

# Predict
res = model.predict(np.expand_dims(sequence, axis=0))
```

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Author
Marwan Gutin
[GitHub: @MarwanGutin](https://github.com/MarwanGutin)


## TODO
 - Add support for numbers and custom phrases
 - Implement feedback game mode
 - Export model to TFLite for mobile deployment


