# 3D Gaussian Splatting – Docker Setup

> Slovenská verzia: [README.md](../README.md)

Complete environment for creating 3D scenes from photographs.

| Component | Description |
|---|---|
| **COLMAP 3.9.1** | Photogrammetry – computes 3D model from photos, GUI and CUDA acceleration |
| **3D Gaussian Splatting** | Training a neural 3D scene representation |
| **SIBR Viewer** | Interactive viewer for trained scenes |

**Detailed instructions:**
- [GUIDE.md](GUIDE.md) – English
- [NAVOD.md](NAVOD.md) – Slovak

---

## Supported platforms

| Platform | Supported | Note |
|---|---|---|
| Ubuntu 20.04 | ✓ | fully working |
| Ubuntu 22.04 | ✓ | fully working, recommended |
| Ubuntu 24.04 | ✓ | fully working |
| Windows 11 + WSL2 | ✓ | run from WSL2 terminal, GUI via WSLg |
| Windows 10 + WSL2 | ✓ | requires VcXsrv for GUI |
| Windows without WSL2 | ✗ | GPU access requires WSL2 |

---

## Requirements

| Requirement | Minimum |
|---|---|
| GPU | NVIDIA, min. 8 GB VRAM |
| RAM | 16 GB |
| Disk | ~30 GB for Docker image |
| Docker | 24+ |
| nvidia-container-toolkit | latest |

---

## Installation

### 1. Docker

**Ubuntu 20.04 / 22.04 / 24.04** – official Docker Inc. repository:
```bash
sudo apt-get update
sudo apt-get install -y ca-certificates curl
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc

sudo tee /etc/apt/sources.list.d/docker.sources <<EOF
Types: deb
URIs: https://download.docker.com/linux/ubuntu
Suites: $(. /etc/os-release && echo "${UBUNTU_CODENAME:-$VERSION_CODENAME}")
Components: stable
Architectures: $(dpkg --print-architecture)
Signed-By: /etc/apt/keyrings/docker.asc
EOF

sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

# Add yourself to the docker group (run without sudo)
sudo usermod -aG docker $USER
newgrp docker
```

Verify:
```bash
docker run hello-world
```

Full documentation: https://docs.docker.com/engine/install/ubuntu/

**Windows 10 / 11:**
- Install [Docker Desktop](https://www.docker.com/products/docker-desktop/)
- In settings enable **WSL2 backend**
- Run all commands from the **WSL2 terminal** (Ubuntu from Microsoft Store)

---

### 2. nvidia-container-toolkit (GPU support for Docker)

**Ubuntu 20.04 / 22.04 / 24.04 + Windows WSL2:**
```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
  sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

---

### 3. Configure data paths

Open the `.env` file and set paths for your system:

```bash
# Linux
DATA_PATH=/home/michal/my_data
OUTPUT_PATH=/home/michal/my_results

# Windows WSL2 (use Linux path format)
DATA_PATH=/mnt/c/Users/Michal/gaussian_data
OUTPUT_PATH=/mnt/c/Users/Michal/gaussian_output
```

Inside the container, data is always available as `data/` and results as `output/`.

---

### 4. Build Docker image

```bash
# One-time build – takes 30–60 minutes
docker compose build
```

> **Windows 10 – build crashes on RAM?** Create `C:\Users\<name>\.wslconfig`:
> ```ini
> [wsl2]
> memory=6GB
> swap=4GB
> ```
> Then restart Docker Desktop.

---

## Running

> **Linux – one-time setup before first run:**  
> If you get `permission denied while trying to connect to the Docker daemon`, add yourself to the `docker` group:
> ```bash
> sudo usermod -aG docker $USER
> newgrp docker   # effective immediately; permanent after next login
> ```

**Linux / WSL2:**
```bash
./run.sh
```

**Windows 10:**
1. Install and launch [VcXsrv](https://sourceforge.net/projects/vcxsrv/) (XLaunch: Multiple windows, Display 0, Disable access control ✓)
2. Run `run.bat`

---

## Test data

The `truck` dataset (Tanks and Temples) is ready to use without COLMAP:

```bash
python3 train.py -s data/truck --data_device cpu
```

---

## Troubleshooting

### `permission denied` on docker commands
```bash
sudo usermod -aG docker $USER
newgrp docker
```

### GUI does not open (`cannot connect to X server`)
```bash
# On the host before running ./run.sh:
xhost +local:docker
```
Windows 11: GUI works automatically via WSLg.  
Windows 10: install and launch [VcXsrv](https://sourceforge.net/projects/vcxsrv/).

### CUDA not available inside the container
```bash
# Test inside the container:
nvidia-smi
python3 -c "import torch; print(torch.cuda.is_available())"
```
If it fails – check nvidia-container-toolkit and restart Docker.

### `CUDA out of memory` during training
```bash
python3 train.py -s data/my_scene --data_device cpu --resolution 2
python3 train.py -s data/my_scene --data_device cpu --iterations 7000
```

### COLMAP reconstruction is empty
- Images overlap too little – every point in the scene must be visible from at least 3 angles
- Try **Exhaustive matching** instead of Sequential
- Recommended number of photos: 50–200

---

## Project structure

```
gaussian_splatting/
├── Dockerfile              ← Docker image definition
├── docker-compose.yml      ← GPU and volume configuration
├── .env                    ← data paths (edit before use)
├── run.sh                  ← container launcher (Linux / WSL2)
├── run.bat                 ← container launcher (Windows)
├── NAVOD.md                ← detailed guide for students (Slovak)
└── GUIDE.md                ← detailed guide for students (English)
```
