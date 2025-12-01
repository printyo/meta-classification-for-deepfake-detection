import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# --- CONFIG ---
# Change this to your local folder where the .npz files are stored
NPZ_DIR = Path(r"C:\Users\admin\Documents\te\datasets\faces_ffpp")   # ← change path here

# Pick the first .npz automatically (or specify one)
#sample = next(NPZ_DIR.glob("*.npz"))
#sample = NPZ_DIR / "original__735.npz"   # manual selection
#sample = NPZ_DIR / "face2face__735_774.npz"   # manual selection
#sample = NPZ_DIR / "faceshifter__735_774.npz"   # manual selection
#sample = NPZ_DIR / "deepfakes__735_774.npz"   # manual selection

#sample = NPZ_DIR / "original__191.npz"   # manual selection
#sample = NPZ_DIR / "face2face__191_188.npz"   # manual selection
#sample = NPZ_DIR / "faceshifter__191_188.npz"   # manual selection
#sample = NPZ_DIR / "deepfakes__191_188.npz"   # manual selection
#sample = NPZ_DIR / "faceswap__191_188.npz"   # manual selection
sample = NPZ_DIR / "neuraltextures__191_188.npz"   # manual selection

print(f"Previewing: {sample}")

# --- LOAD ---
data = np.load(sample)
key = "frames" if "frames" in data else "faces"
frames = data[key]   # shape (8,224,224,3)
print(f"Shape: {frames.shape}, Key: {key}")

# --- CONVERT BGR → RGB ---
frames_rgb = frames[..., ::-1]   # swap channel order

# --- DISPLAY ---
n = min(8, len(frames_rgb))
plt.figure(figsize=(16, 4))
for i in range(n):
    plt.subplot(1, n, i + 1)
    plt.imshow(frames_rgb[i])
    plt.axis("off")
plt.tight_layout()
plt.show()
