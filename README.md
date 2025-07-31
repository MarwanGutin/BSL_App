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
─── marwangutin-bsl_app/
    ├── best_model.h5
    ├── dataset.py
    ├── LICENSE
    ├── main.py
    ├── model.py
    ├── mp_files.zip
    ├── my_classes.py
    └── utils.py


---

## 🚀 Getting Started

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

🟩 HELLO   |███████████████████
🟧 YES     |██████
🟥 THANKS  |█
🤖 Model Architecture
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
## 📄 License
This project is licensed under the MIT License. See the LICENSE file for details.

## 👤 Author
Marwan Gutin
GitHub: @MarwanGutin


