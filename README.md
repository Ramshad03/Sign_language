# ASL Recognition System

Real-time American Sign Language (ASL) recognition using MediaPipe hand tracking, a PyTorch MLP classifier, and a hold-to-confirm sentence builder with text-to-speech output.

## How it works

1. A webcam captures live video.
2. MediaPipe extracts 21 hand landmarks (63 coordinates: x, y, z per landmark).
3. Landmarks are normalised relative to the wrist and scaled to [-1, 1].
4. A TorchScript MLP predicts one of 29 gestures (A–Z + SPACE, DEL, NOTHING).
5. A majority-vote smoother (window = 7 frames) stabilises noisy predictions.
6. A hold-to-confirm state machine confirms a gesture only when held for 1 second.
7. Confirmed gestures build words; SPACE commits the word and speaks it via Windows SAPI5 TTS.

## Project structure

```
Sign_lang/
├── app.py               # Main launcher — camera + inference + HUD + TTS
├── collect_data.py      # Step 1: record 100 landmark samples per gesture
├── build_dataset.py     # Step 2: merge .npy files → dataset.npz + labels.json
├── preprocess.py        # Step 3: normalise + augment → dataset_augmented.npz
├── train_model.py       # Step 4: train ASLNet MLP, save best checkpoint
├── evaluate_model.py    # Step 5: confusion matrix, per-class accuracy, export
├── inference.py         # Standalone real-time inference (no sentence builder)
├── sentence_builder.py  # Sentence builder module (used by app.py)
├── env_check.py         # Dependency / environment sanity check
├── data/                # Collected landmark files (one folder per gesture)
└── model/               # Trained weights, labels.json, training curve PNG
```

## Requirements

- Python 3.10+
- Windows (TTS uses Windows SAPI5 via `win32com`)
- CUDA-capable GPU recommended (falls back to CPU automatically)

Install dependencies:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install mediapipe opencv-python numpy matplotlib seaborn scikit-learn pywin32
```

## Training pipeline

Run these scripts in order to train from scratch:

```bash
# 1. Collect landmark data (100 samples × 29 gestures)
python collect_data.py

# 2. Build dataset.npz and labels.json
python build_dataset.py

# 3. Normalise and augment
python preprocess.py

# 4. Train the MLP (80 epochs, saves best checkpoint)
python train_model.py

# 5. Evaluate and export TorchScript model for inference
python evaluate_model.py
```

## Running the app

```bash
python app.py
```

### Keyboard controls

| Key | Action |
|-----|--------|
| `S` | Speak the full sentence aloud |
| `C` | Clear the entire sentence |
| `Q` | Quit |

### Gesture controls (hold for 1 second to confirm)

| Gesture | Action |
|---------|--------|
| `A`–`Z` | Append letter to current word |
| `SPACE` | Commit current word and speak it |
| `DEL` | Delete last letter (or last word if word is empty) |

## Model architecture

```
ASLNet (MLP)
  Input:  63 features  (21 landmarks × 3 normalised coords)
  Layer 1: Linear(63 → 512) → BatchNorm → ReLU → Dropout(0.4)
  Layer 2: Linear(512 → 256) → BatchNorm → ReLU → Dropout(0.4)
  Layer 3: Linear(256 → 128) → BatchNorm → ReLU → Dropout(0.4)
  Output:  Linear(128 → 29)
```

Training uses Adam (lr=1e-3, weight decay=1e-4) with `ReduceLROnPlateau` scheduling and CrossEntropyLoss over 80 epochs. The best checkpoint by test accuracy is saved automatically.

## Configuration

Key constants in `app.py`:

| Constant | Default | Description |
|----------|---------|-------------|
| `CONFIDENCE_THRESH` | 0.85 | Minimum softmax confidence to accept a prediction |
| `SMOOTH_WINDOW` | 7 | Majority-vote window size (frames) |
| `HOLD_SECONDS` | 1.0 | Seconds a gesture must be held to confirm |
| `COOLDOWN_SECONDS` | 0.5 | Lockout period after a confirmation |
| `CAMERA_INDEX` | 0 | Webcam index |


## For Further updation 

To retrain the letter:-
 1. Remove that letter: 'Remove-Item -Recurse -Force "C:\Users\muham\OneDrive\Desktop\Work\Sign_lang\data\X"'
 2. Record only that letter: 'python collect_data.py'
 3. Rebuild and Retrain: 'python build_dataset.py
                          python preprocess.py
                          python train_model.py
                          python evaluate_model.py'
 4. Launch app again: 'python app.py'

 ## Run venv
.\venv\Scripts\Activate.ps1

 ## Run the project 
    python app.py 

## Run harvesting pipeline

# 1. Letters (skip J and Z — already excluded)
python collect_data.py

# 2. Words + J + Z together
python collect_word_data.py

# 3. Build and train both models
python build_dataset.py
python preprocess.py
python train_model.py

python word_dataset_builder.py
python word_train.py

# 4. Run
python app.py

______________________________________________________________________________________________________________________________


#### TO TRAIN
python collect_data.py         # adds 100 MORE samples per letter
python build_dataset.py
python preprocess.py
python train_model.py
python evaluate_model.py

python collect_word_data.py    # adds 60 MORE segments per word
python word_dataset_builder.py
python word_train.py

python app.py

_______________________________________________________________________________________________________________________________

##### TO Re-TRAIN SPECIFIC

# Collect only X and C (problem letters)
python collect_data.py X C

# Collect only Z and hello (problem words)
python collect_word_data.py Z hello 

# Collect multiple specific ones
python collect_data.py X C K S

# Collect everything (no args = same as before)
python collect_data.py
python collect_word_data.py

# After adding more letter samples:
python build_dataset.py
python preprocess.py
python train_model.py
python evaluate_model.py

# After adding more word/J/Z samples:
python word_dataset_builder.py
python word_train.py

_______________________________________________________________________________________________________________________________

### TO DELETE SPECIFIC DATA


# Delete X data and collect fresh 100 samples
python collect_data.py --clear X

# Delete X and C, then collect both fresh
python collect_data.py --clear X C

# Delete Z data and collect fresh Z segments
python collect_word_data.py --clear Z

# Delete hello and bad, collect both fresh
python collect_word_data.py --clear hello bad

# And then save that gesture
python collect_word_data.py --clear please
python word_dataset_builder.py
python word_train.py

________________________________________________________________________________________________________________________________

## TO ADD NEW WORDS

# Collect only the new word (no need to redo all words)
python collect_word_data.py sorry water

# Rebuild dataset (automatically includes all existing + new data)
python word_dataset_builder.py

# Retrain
python word_train.py

# Run
python app.py

_________________________________________________________________________________________________________________________________
_________________________________________________________________________________________________________________________________

## Two ways to train — side by side

# Method 1 — Manual (webcam)
python collect_data.py          # or: python collect_data.py X C
python collect_word_data.py     # or: python collect_word_data.py hello Z

python build_dataset.py && python preprocess.py && python train_model.py && python evaluate_model.py
python word_dataset_builder.py && python word_train.py

# Method 2 — Harvesting (from videos)
raw_videos/
  A/         ← put letter A videos here (.mp4/.avi/.mov)
  B/
  hello/     ← put hello word videos here
  thanks/

python harvest.py               # processes everything automatically

# or specific ones:
python harvest.py hello A bad

# Then train:
python build_dataset.py; python preprocess.py; python train_model.py; python evaluate_model.py
python word_dataset_builder.py; python word_train.py


# Both methods add data on top of each other — you can mix manual + harvested data and retrain. The more data from both sources, the better the model.
 
 ________________________________________________________________________________________________________________________________
 ________________________________________________________________________________________________________________________________
 


 #### TO ADD NEW LETTERS AND WORDS


 
# Static letter (e.g. G, new custom sign)

1. Add it to collect_data.py:

ALL_GESTURES = (
    list('ABCDEFHIKLMNOPQRSTUVWXY') + ['NEW_LETTER'] +  # ← add here
    ['SPACE', 'DEL', 'NOTHING']
)


2. Collect + retrain:

python collect_data.py NEW_LETTER
python build_dataset.py; python preprocess.py; python train_model.py; python evaluate_model.py



# Motion letter (drawn gesture, like J or Z)

1. Add it to collect_word_data.py:

ALL_GESTURES = [
    "J", "Z", "P", "Q", "NEW_LETTER",   # ← add here
    ...
]


2. Collect + retrain:

python collect_word_data.py NEW_LETTER
python word_dataset_builder.py; python word_train.py



# to add to git

git remote add origin https://github.com/YOUR_USERNAME/Sign_lang.git
git push -u origin master



 .\venv\Scripts\Activate.ps1     