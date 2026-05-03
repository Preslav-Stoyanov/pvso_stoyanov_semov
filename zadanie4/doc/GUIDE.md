# 3D Gaussian Splatting – Student Guide

> Slovenská verzia: [NAVOD.md](NAVOD.md) | [README](../README.md)

## Before you start: capturing your own dataset

You can use the [ximea-test.py](../ximea-test.py) script to capture images for your own dataset. It also includes camera configuration options.

A note on **Automatic vs fixed settings**:
- **Automatic** (exposure/gain/white balance) is a quick start, but brightness and colour change between frames, which hurts data consistency.
- **Fixed** values keep the appearance consistent across all photos, which typically leads to more stable COLMAP reconstruction and better 3DGS training.

Recommendation: do a quick test in automatic mode, then switch to fixed values for the full dataset capture.

## Pipeline

```
┌─────────────────────────────────────────────────────────┐
│  DATA PREPARATION (COLMAP)                              │
│                                                         │
│  1. Create project   →  set paths in COLMAP             │
│  2. Feature Extract  →  detect keypoints                │
│  3. Feature Matching →  match points across images      │
│  4. Reconstruction   →  compute 3D model + cameras      │
│  5. Export model     →  save to distorted/              │
│  6. Undistortion     →  convert.py corrects distortion  │
└───────────────────────┬─────────────────────────────────┘
                        │
                        ▼
┌───────────────────────────────────────────────────────┐
│  TRAINING                                             │
│                                                       │
│  python3 train.py -s data/my_scene                    │
│         --data_device cpu                             │
│                                                       │
│  ~30–60 minutes, result: output/<uuid>/               │
└───────────────────────┬───────────────────────────────┘
                        │
                        ▼
┌───────────────────────────────────────────────────────┐
│  VIEWING                                              │
│                                                       │
│  SIBR_gaussianViewer_app -m output/<uuid>             │
└───────────────────────────────────────────────────────┘
```

---

## Requirements

- Docker (with GPU access)
- NVIDIA GPU (min. 8 GB VRAM)
- Linux or Windows 11 with WSL2 (Windows 10 works but requires VcXsrv for GUI)
- Min. 8 GB RAM (16 GB recommended) – COLMAP build is memory-intensive

---

## Before starting – configure paths

Open the `.env` file and set where your data is:

```
DATA_PATH=/path/to/your/data
OUTPUT_PATH=/path/where/results/are/saved
```

Example:
```
DATA_PATH=/home/michal/Desktop/gaussian-splatting/data
OUTPUT_PATH=/home/michal/Desktop/gaussian-splatting/output
```

---

## Starting the container

**Linux / WSL2:**
```bash
./run.sh
```

**Windows 10:**
1. Install [VcXsrv](https://sourceforge.net/projects/vcxsrv/) and launch XLaunch: Multiple windows, Display 0, Disable access control ✓
2. Run `run.bat`

> **Windows 10 – build crashes on RAM?** Create `C:\Users\<name>\.wslconfig`:
> ```ini
> [wsl2]
> memory=6GB
> swap=4GB
> ```
> Then restart Docker Desktop.

All further commands are run **inside the container**.

---

## Part 1 – Data preparation with COLMAP

### Folder structure before starting

Create a folder for your scene and put your photos in it:

```
data/
└── my_scene/
    └── input/       ← PUT YOUR PHOTOS HERE (JPG or PNG)
```

> Photos must be in the `input/` folder – not `images/` or anywhere else.

---

### Launch COLMAP GUI

```bash
colmap gui
```

---

### Step 1 – Create project

- **File → New project**
- **Database** – set the database path:
  ```
  /workspace/data/my_scene/sparse/database.db
  ```
- **Images** – set the image path:
  ```
  /workspace/data/my_scene/input
  ```

> The `sparse/` folder must exist. If not, create it before opening COLMAP:
> ```bash
> mkdir -p /workspace/data/my_scene/sparse
> ```

- Click **Save**

---

### Step 2 – Feature Extraction

COLMAP detects characteristic points (corners, edges) in each photo.

- **Processing → Feature extraction**
- Camera model: change to **SIMPLE_PINHOLE**
- Click **Extract** and wait

---

### Step 3 – Feature Matching

COLMAP matches points across photos – finds which points correspond to the same 3D location.

- **Processing → Feature matching**
  - Up to 100 images: choose **Exhaustive**
  - More images (sequential): choose **Sequential**
- Click **Run** and wait

---

### Step 4 – Reconstruction

From the matched points, COLMAP computes camera positions and creates a 3D point cloud.

- **Reconstruction → Start reconstruction**
- Watch cameras and points appear in the 3D view
- Wait until it finishes

---

### Step 5 – Export model

Export the model to `distorted/sparse/0/` – this is where `convert.py` expects it.

First create the folder:
```bash
mkdir -p /workspace/data/my_scene/distorted/sparse/0
```

Then in COLMAP:
- **File → Export model** ← exactly this, NOT "Export model as text"!
- Save to: `/workspace/data/my_scene/distorted/sparse/0`
- Click **Export**

This creates: `cameras.bin`, `images.bin`, `points3D.bin`

You can close COLMAP now.

---

### Folder structure after COLMAP export

```
data/
└── my_scene/
    ├── input/                        ← your original photos
    ├── sparse/
    │   └── database.db               ← COLMAP database (created automatically)
    └── distorted/
        └── sparse/
            └── 0/
                ├── cameras.bin       ← exported model
                ├── images.bin
                └── points3D.bin
```

---

### Step 6 – Undistortion (convert.py)

`convert.py` corrects lens distortion and prepares data for training:
- takes photos from `input/`
- takes model from `distorted/sparse/0/`
- creates undistorted photos in `images/`
- copies model to `sparse/0/`

```bash
python3 convert.py -s data/my_scene --skip_matching
```

> **Important:** Without this step `train.py` will crash with `Could not recognize scene type!` — the `images/` and `sparse/0/` folders must exist before training.

---

### Final folder structure (ready for training)

```
data/
└── my_scene/
    ├── input/                        ← original photos (unchanged)
    ├── images/                       ← undistorted photos (created by convert.py)
    ├── sparse/
    │   ├── database.db
    │   └── 0/
    │       ├── cameras.bin           ← copied by convert.py from distorted/
    │       ├── images.bin
    │       └── points3D.bin
    └── distorted/
        └── sparse/
            └── 0/
                ├── cameras.bin       ← exported from COLMAP
                ├── images.bin
                └── points3D.bin
```

---

## Part 2 – Training

```bash
python3 train.py -s data/my_scene --data_device cpu
```

At the start, the output folder is printed:
```
Output folder: ./output/a1b2c3d4-5
```

**Optional parameters:**
```bash
# Faster training (lower quality)
python3 train.py -s data/my_scene --data_device cpu --iterations 7000

# Low VRAM – reduce resolution by 2x
python3 train.py -s data/my_scene --data_device cpu --resolution 2
```

Training takes **20–60 minutes**. Progress is printed to the terminal (loss every 100 iterations).

---

## Part 3 – Viewing results

Find the UUID of the output folder:
```bash
ls output/
```

Launch the viewer:
```bash
SIBR_gaussianViewer_app -m output/<uuid>
```

### Viewer controls

| Action | Control |
|---|---|
| Move | `W` `A` `S` `D` |
| Move up/down | `Q` `E` |
| Rotate | Right mouse button + drag |
| Zoom | Mouse wheel |
| Movement speed | `+` / `-` |
| Close | `ESC` |

---

## Test data – Truck dataset

This dataset already has `images/` and `sparse/0/` prepared, so you can skip Part 1:

```bash
python3 train.py -s data/truck --data_device cpu
```

---

## Common errors

**`cameras, images, points3D files do not exist at distorted/sparse/0`**
- You forgot to export the model (Step 5), or exported to the wrong folder

**`Export model as text` instead of `Export model`**
- `convert.py` needs `.bin` files – export via **File → Export model** (not "as text")

**`CUDA out of memory`**
- Add `--resolution 2` or `--iterations 7000`

**`Could not recognize scene type!`**
- `convert.py` was not run – `images/` or `sparse/0/` is missing in the scene folder
- Check the folder structure in the "Final folder structure" section above

**COLMAP GUI does not open**
- Linux: Run on the host: `xhost +local:docker`
- Windows 10: Launch VcXsrv (XLaunch) before the container, Display number = 0, Disable access control ✓
- Make sure the container was started via `./run.sh` / `run.bat`
