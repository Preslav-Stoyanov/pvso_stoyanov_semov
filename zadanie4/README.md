# 3D Gaussian Splatting – Docker Setup

> English version: [doc/README_EN.md](doc/README_EN.md)

Kompletné prostredie pre tvorbu 3D scén z fotografií.

| Komponent | Popis |
|---|---|
| **COLMAP 3.9.1** | Fotogrametria – výpočet 3D modelu z fotografií, GUI aj CUDA akcelerácia |
| **3D Gaussian Splatting** | Trénovanie neurálnej 3D reprezentácie scény |
| **SIBR Viewer** | Interaktívne prezeranie natrénovaných scén |

**Podrobný postup práce:**
- [NAVOD.md](doc/NAVOD.md) – slovensky
- [GUIDE.md](doc/GUIDE.md) – English

---

## Podporované platformy

| Platforma | Podporované | Poznámka |
|---|---|---|
| Ubuntu 20.04 | ✓ | plne funkčné |
| Ubuntu 22.04 | ✓ | plne funkčné, odporúčané |
| Ubuntu 24.04 | ✓ | plne funkčné |
| Windows 11 + WSL2 | ✓ | spúšťaj z WSL2 terminálu, GUI cez WSLg |
| Windows 10 + WSL2 | ✓ | potrebuje VcXsrv pre GUI |
| Windows bez WSL2 | ✗ | GPU prístup vyžaduje WSL2 |

---

## Požiadavky

| Požiadavka | Minimum |
|---|---|
| GPU | NVIDIA, min. 8 GB VRAM |
| RAM | 16 GB |
| Disk | ~30 GB pre Docker image |
| Docker | 24+ |
| nvidia-container-toolkit | latest |

---

## Inštalácia

### 1. Docker

**Ubuntu 20.04 / 22.04 / 24.04** – oficiálny repozitár Docker Inc.:
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

# Pridaj seba do docker skupiny (bez sudo)
sudo usermod -aG docker $USER
newgrp docker
```

Verifikácia:
```bash
docker run hello-world
```

Plná dokumentácia: https://docs.docker.com/engine/install/ubuntu/

**Windows 10 / 11:**
- Nainštaluj [Docker Desktop](https://www.docker.com/products/docker-desktop/)
- V nastaveniach zapni **WSL2 backend**
- Všetky príkazy spúšťaj z **WSL2 terminálu** (Ubuntu z Microsoft Store)

---

### 2. nvidia-container-toolkit (GPU podpora pre Docker)

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

### 3. Nastavenie ciest k dátam

Otvor súbor `.env` a nastav cesty podľa tvojho systému:

```bash
# Linux
DATA_PATH=/home/michal/moje_data
OUTPUT_PATH=/home/michal/moje_vysledky

# Windows WSL2 (cesta v linuxovom formáte)
DATA_PATH=/mnt/c/Users/Michal/gaussian_data
OUTPUT_PATH=/mnt/c/Users/Michal/gaussian_output
```

Vo vnútri kontajnera sú dáta vždy dostupné ako `data/` a výsledky ako `output/`.

---

### 4. Build Docker image

```bash
# Jednorazový build – trvá 30–60 minút
docker compose build
```

---

## Spustenie

> **Linux – jednorazové nastavenie pred prvým spustením:**  
> Ak dostaneš chybu `permission denied while trying to connect to the Docker daemon`, pridaj sa do skupiny `docker`:
> ```bash
> sudo usermod -aG docker $USER
> newgrp docker   # platí pre aktuálnu session; pri ďalšom prihlásení automaticky
> ```

```bash
./run.sh
```

Na Windows spusti z WSL2 terminálu. Na Windows 10 spusti pred tým [VcXsrv](https://sourceforge.net/projects/vcxsrv/) pre GUI.

---

## Testovacie dáta

Dataset `truck` (Tanks and Temples) je pripravený bez nutnosti COLMAP:

```bash
python3 train.py -s data/truck --data_device cpu
```

---

## Riešenie problémov

### `permission denied` pri docker príkazoch
```bash
sudo usermod -aG docker $USER
newgrp docker
```

### GUI sa neotvára (`cannot connect to X server`)
```bash
# Na hoste pred spustením ./run.sh:
xhost +local:docker
```
Na Windows 11: GUI funguje automaticky cez WSLg.  
Na Windows 10: nainštaluj a spusti [VcXsrv](https://sourceforge.net/projects/vcxsrv/).

### CUDA nie je dostupná v kontajneri
```bash
# Otestuj vo vnútri kontajnera:
nvidia-smi
python3 -c "import torch; print(torch.cuda.is_available())"
```
Ak nefunguje – skontroluj nvidia-container-toolkit a reštartuj Docker.

### `CUDA out of memory` pri trénovaní
```bash
python3 train.py -s data/moja_scena --data_device cpu --resolution 2
python3 train.py -s data/moja_scena --data_device cpu --iterations 7000
```

### COLMAP rekonštrukcia je prázdna
- Fotky sa málo prekrývajú – každý bod scény musí byť viditeľný aspoň z 3 uhlov
- Skús **Exhaustive matching** namiesto Sequential
- Odporúčaný počet fotiek: 50–200

---

## Štruktúra projektu

```
gaussian_splatting/
├── Dockerfile              ← definícia Docker image
├── docker-compose.yml      ← konfigurácia GPU a mountov
├── .env                    ← cesty k dátam (uprav pred použitím)
├── run.sh                  ← spúšťač kontajnera (Linux / WSL2)
├── run.bat                 ← spúšťač pre Windows
└── doc/
    ├── NAVOD.md            ← podrobný postup pre študentov (SK)
    ├── GUIDE.md            ← podrobný postup pre študentov (EN)
    └── README_EN.md        ← anglická verzia README
```
