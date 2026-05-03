# 3D Gaussian Splatting – Návod pre študentov

> English version: [GUIDE.md](GUIDE.md) | [README](../README.md)

## Pred začiatkom: snímanie vlastného datasetu

Na snímanie obrazov pre vlastný dataset môžeš použiť skript [ximea-test.py](../ximea-test.py), kde sú aj možnosti nastavenia kamery.

Krátko k nastaveniam **Automatic vs pevne nastavené**:
- **Automatic** (expozícia/gain/white balance) je rýchly štart, ale medzi zábermi sa mení jas a farba, čo zhoršuje konzistenciu dát.
- **Pevne nastavené** hodnoty držia rovnaký vzhľad všetkých fotiek, čo zvyčajne vedie k stabilnejšej rekonštrukcii v COLMAP a lepšiemu tréningu 3DGS.

Odporúčanie: po krátkom teste v automatic režime prepnúť na pevné hodnoty a s nimi nasnímať celý dataset.

## Pipeline

```
┌─────────────────────────────────────────────────────────┐
│  PRÍPRAVA DÁT (COLMAP)                                  │
│                                                         │
│  1. Vytvorenie projektu  →  nastavenie ciest v COLMAP   │
│  2. Feature Extraction   →  nájdenie kľúčových bodov    │
│  3. Feature Matching     →  párovanie bodov medzi fotkami│
│  4. Reconstruction       →  výpočet 3D modelu a kamier  │
│  5. Export modelu        →  uloženie do distorted/       │
│  6. Undistortion         →  convert.py opraví distorziu │
└───────────────────────┬─────────────────────────────────┘
                        │
                        ▼
┌───────────────────────────────────────────────────────┐
│  TRÉNOVANIE                                           │
│                                                       │
│  python3 train.py -s data/moja_scena                  │
│         --data_device cpu                             │
│                                                       │
│  ~30–60 minút, výsledok: output/<uuid>/               │
└───────────────────────┬───────────────────────────────┘
                        │
                        ▼
┌───────────────────────────────────────────────────────┐
│  ZOBRAZENIE                                           │
│                                                       │
│  SIBR_gaussianViewer_app -m output/<uuid>             │
└───────────────────────────────────────────────────────┘
```

---

## Požiadavky

- Docker (s prístupom k GPU)
- NVIDIA GPU (min. 8 GB VRAM)
- Linux alebo Windows 11 s WSL2 (Windows 10 funguje, ale vyžaduje VcXsrv pre GUI)
- Min. 8 GB RAM (odporúčané 16 GB) – build COLMAP je pamäťovo náročný

---

## Pred spustením – nastavenie ciest

Otvor súbor `.env` a nastav kde máš dáta:

```
DATA_PATH=/cesta/k/priečinku/data
OUTPUT_PATH=/cesta/kde/sa/ulozia/vysledky
```

Príklad:
```
DATA_PATH=/home/michal/Desktop/gaussian-splatting/data
OUTPUT_PATH=/home/michal/Desktop/gaussian-splatting/output
```

---

## Spustenie kontajnera

**Linux/WSL2:**
```bash
./run.sh
```

**Windows 10:**
1. Nainštaluj [VcXsrv](https://sourceforge.net/projects/vcxsrv/) a spusti XLaunch: Multiple windows, Display 0, Disable access control ✓
2. Spusti `run.bat`

> **Windows 10 – build zlyhá na RAM?** Vytvor `C:\Users\<meno>\.wslconfig`:
> ```ini
> [wsl2]
> memory=6GB
> swap=4GB
> ```
> Potom reštartuj Docker Desktop.

Všetky ďalšie príkazy píš **vo vnútri kontajnera**.

---

## Časť 1 – Príprava dát pomocou COLMAP

### Štruktúra priečinka pred začatím

Vytvor priečinok pre scénu a daj tam fotky:

```
data/
└── moja_scena/
    └── input/       ← SEM daj svoje fotky (JPG alebo PNG)
```

> Fotky musia byť v priečinku `input/` – nie `images/` ani inde.

---

### Spustenie COLMAP GUI

```bash
colmap gui
```

---

### Krok 1 – Vytvorenie projektu

- **File → New project**
- **Database** – nastav cestu k databáze:
  ```
  /workspace/data/moja_scena/sparse/database.db
  ```
- **Images** – nastav cestu k fotkám:
  ```
  /workspace/data/moja_scena/input
  ```

> Priečinok `sparse/` musí existovať. Ak neexistuje, vytvor ho pred spustením COLMAP:
> ```bash
> mkdir -p /workspace/data/moja_scena/sparse
> ```

- Klikni **Save**

---

### Krok 2 – Feature Extraction

COLMAP nájde charakteristické body (rohy, hrany) na každej fotografii.

- **Processing → Feature extraction**
- Camera model: zmeň na **SIMPLE_PINHOLE**
- Klikni **Extract** a počkaj

---

### Krok 3 – Feature Matching

COLMAP spáruje body medzi fotografiami – zistí ktoré body zodpovedajú rovnakému miestu v 3D.

- **Processing → Feature matching**
  - Do 100 fotiek: zvol **Exhaustive**
  - Viac fotiek (po sebe idúce): zvol **Sequential**
- Klikni **Run** a počkaj

---

### Krok 4 – Rekonštrukcia

Z nájdených zhôd COLMAP vypočíta pozície kamier a vytvorí 3D mrak bodov.

- **Reconstruction → Start reconstruction**
- Uvidíš ako sa v 3D okne objavujú kamery a body
- Počkaj kým sa dokončí

---

### Krok 5 – Export modelu

Model exportuj do priečinka `distorted/sparse/0/` – tam ho očakáva `convert.py`.

Najprv vytvor priečinok:
```bash
mkdir -p /workspace/data/moja_scena/distorted/sparse/0
```

Potom v COLMAP:
- **File → Export model** ← presne toto, nie "Export model as text"!
- Ulož do: `/workspace/data/moja_scena/distorted/sparse/0`
- Klikni **Export**

Vzniknú súbory: `cameras.bin`, `images.bin`, `points3D.bin`

COLMAP môžeš zatvoriť.

---

### Štruktúra po exporte z COLMAP

```
data/
└── moja_scena/
    ├── input/                        ← tvoje originálne fotky
    ├── sparse/
    │   └── database.db               ← COLMAP databáza (vytvorená automaticky)
    └── distorted/
        └── sparse/
            └── 0/
                ├── cameras.bin       ← exportovaný model
                ├── images.bin
                └── points3D.bin
```

---

### Krok 6 – Undistortion (convert.py)

`convert.py` opraví distorziu objektívu a pripraví dáta pre trénovanie:
- vezme fotky z `input/`
- vezme model z `distorted/sparse/0/`
- vytvorí undistortované fotky do `images/`
- skopíruje model do `sparse/0/`

```bash
python3 convert.py -s data/moja_scena --skip_matching
```

> **Dôležité:** Bez tohto kroku `train.py` spadne s chybou `Could not recognize scene type!` — priečinok `images/` a `sparse/0/` musia existovať pred trénovaním.

---

### Štruktúra po convert.py (finálna, pripravená na trénovanie)

```
data/
└── moja_scena/
    ├── input/                        ← originálne fotky (nezmenené)
    ├── images/                       ← undistortované fotky (vytvoril convert.py)
    ├── sparse/
    │   ├── database.db
    │   └── 0/
    │       ├── cameras.bin           ← skopíroval convert.py z distorted/
    │       ├── images.bin
    │       └── points3D.bin
    └── distorted/
        └── sparse/
            └── 0/
                ├── cameras.bin       ← export z COLMAP
                ├── images.bin
                └── points3D.bin
```

---

## Časť 2 – Trénovanie

```bash
python3 train.py -s data/moja_scena --data_device cpu
```

Na začiatku sa vypíše kam sa ukladajú výsledky:
```
Output folder: ./output/a1b2c3d4-5
```

**Voliteľné parametre:**
```bash
# Rýchlejšie trénovanie (nižšia kvalita)
python3 train.py -s data/moja_scena --data_device cpu --iterations 7000

# Ak máš málo VRAM – zmenši rozlíšenie 2x
python3 train.py -s data/moja_scena --data_device cpu --resolution 2
```

Trénovanie trvá **20–60 minút**. Priebeh vidíš v termináli (loss každých 100 iterácií).

---

## Časť 3 – Zobrazenie výsledkov

Zisti UUID výstupného priečinka:
```bash
ls output/
```

Spusti viewer:
```bash
SIBR_gaussianViewer_app -m output/<uuid>
```

### Ovládanie viewera

| Akcia | Ovládanie |
|---|---|
| Pohyb | `W` `A` `S` `D` |
| Pohyb hore/dole | `Q` `E` |
| Rotácia | Pravé tlačidlo myši + ťahanie |
| Zoom | Koliesko myši |
| Rýchlosť pohybu | `+` / `-` |
| Zavrieť | `ESC` |

---

## Testovacie dáta – Truck dataset

Dataset je už pripravený (má hotové `images/` aj `sparse/0/`), preskočíš Časť 1:

```bash
python3 train.py -s data/truck --data_device cpu
```

---

## Časté chyby

**`cameras, images, points3D files do not exist at distorted/sparse/0`**
- Zabudol si exportovať model (Krok 5), alebo si exportoval do zlého priečinka

**`Export model as text` namiesto `Export model`**
- `convert.py` potrebuje `.bin` súbory – exportuj cez **File → Export model** (nie "as text")

**`CUDA out of memory`**
- Pridaj `--resolution 2` alebo `--iterations 7000`

**`Could not recognize scene type!`**
- `convert.py` nebol spustený – chýba `images/` alebo `sparse/0/` v priečinku scény
- Skontroluj štruktúru priečinka podľa sekcie "Štruktúra po convert.py"

**COLMAP GUI sa neotvára**
- Linux: Na hoste spusti: `xhost +local:docker`
- Windows 10: Spusti VcXsrv (XLaunch) pred kontajnerom, Display number = 0, Disable access control ✓
- Skontroluj že kontajner bol spustený cez `./run.sh` / `run.bat`
