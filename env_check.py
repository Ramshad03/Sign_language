# check_env.py — Run this to verify all packages are installed correctly

import sys
print(f"Python        : {sys.version}")

import torch
print(f"PyTorch       : {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU           : {torch.cuda.get_device_name(0)}")
    print(f"VRAM          : {torch.cuda.get_device_properties(0).total_memory // 1024**2} MB")

import cv2
print(f"OpenCV        : {cv2.__version__}")

import mediapipe as mp
print(f"MediaPipe     : {mp.__version__}")

import numpy as np
print(f"NumPy         : {np.__version__}")

import sklearn
print(f"scikit-learn  : {sklearn.__version__}")

import matplotlib
print(f"Matplotlib    : {matplotlib.__version__}")

import seaborn
print(f"Seaborn       : {seaborn.__version__}")

import tqdm
print(f"tqdm          : {tqdm.__version__}")

import pyttsx3
print(f"pyttsx3       : {pyttsx3.__version__}")

print("\n✅ All packages verified successfully!")