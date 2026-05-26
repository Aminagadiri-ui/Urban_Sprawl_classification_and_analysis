# 🛰️ Urban Sprawl Analysis & Classification

> **Multi-City · Multi-Year · Random Forest Classification Pipeline**
> Remote Sensing · Urban Analytics · Google Earth Engine

---

## Project Overview

This project presents a **complete, reproducible, end-to-end pipeline** for urban sprawl analysis using freely available Landsat satellite imagery and Google Earth Engine (GEE). The pipeline classifies urban land cover across **four globally diverse cities** over **six epochs (2000–2025)** and computes eight urban expansion metrics, culminating in the UN **SDG 11.3.1 Land Consumption Rate**.

| Attribute | Details |
|---|---|
| **Study Cities** | Bangkok · Atlanta · Paris · Mexico City |
| **Time Period** | 2000 · 2005 · 2010 · 2015 · 2020 · 2025 |
| **Satellite Sensor** | Landsat 5 / 7 / 8 / 9 — 30 m Surface Reflectance |
| **Classifier** | Random Forest — 200 trees — 9 Spectral Indices |
| **Validation Reference** | GHSL P2023A — JRC / European Commission |
| **SDG Indicator** | 11.3.1 — Land Consumption Rate (LCR) |
| **Platform** | Google Earth Engine + Python (Google Colab) |
| **Team** | Amina Gadiri · Serine Boutiche · Sonia Cherbal · Hibaterrahman Remmache |
| **Group** | Team 4 · Group G7 · National High School of Artificial Intelligence |

---

## Repository Structure

```
urban-sprawl-analysis/
│
├── Urban_Sprawl_Analysis_and_classification_and_Visualisation.ipynb  # Main notebook
│
├── bands_ndvi_bui/               # Exported spectral index rasters (NDVI, NDBI, SWIR)
│
├── classification/               # Exported binary classification GeoTIFFs
│
├── Dashboard/
│   ├── dynamic_urban_sprawl.html # Live-data dashboard (connects to Flask backend)
│   └── static_urban_sprawl.html  # Static snapshot dashboard
│
├── data/
│   ├── Binary_Cairo_Training.csv        # Binary (Urban/Non-Urban) training samples
│   └── Categorical_Cairo_Training.csv   # 4-class training samples (Urban/Veg/Water/Soil)
│
├── results/
│   ├── classification_results_all_cities.csv  # RF urban fractions per city-year
│   └── truth_values_all_cities.csv            # GHSL ground-truth fractions per city-year
│
├── .gitignore
├── LICENSE
├── README.md                     # ← you are here
└── requirements.txt
```

---

## Prerequisites

Before cloning and running this project, ensure you have the following installed and configured on your machine.

### System Requirements

- **Operating System:** Windows 10/11, macOS 12+, or Ubuntu 20.04+
- **RAM:** Minimum 8 GB (16 GB recommended for Phase 7 raster operations)
- **Disk:** At least 5 GB free for exported GeoTIFFs and intermediate files
- **Internet:** Required throughout — GEE operations execute on Google's servers

### Accounts Required

You need **two free accounts** before running anything:

1. **Google Account** with **Google Earth Engine access**
   - Sign up at [https://earthengine.google.com/signup](https://earthengine.google.com/signup)
   - Approval can take 24–48 hours for new accounts
   - You will need your **GEE Cloud Project ID** (created automatically or manually at [console.cloud.google.com](https://console.cloud.google.com))

2. **Google Drive** (included with your Google Account)
   - Used to store exported GeoTIFF files from Phase 6
   - Recommended free space: at least 10 GB

### Software Requirements

| Tool | Version | Purpose |
|---|---|---|
| Python | 3.9 or later | Runtime for all non-GEE phases |
| Google Colab | (browser-based) | Recommended execution environment |
| Git | 2.x | Cloning the repository |

> **Note:** The notebook is designed to run on **Google Colab**, which provides a pre-configured Python environment and simplifies GEE authentication. Running locally requires additional setup (see [Local Execution](#optional-running-locally-outside-colab) below).

---

## Step-by-Step Setup Instructions

Follow every step in order. Do not skip ahead — each phase depends on the one before it.

---

### Step 1 — Clone the Repository

Open a terminal (or Git Bash on Windows) and run:

```bash
git clone https://github.com/<your-username>/urban-sprawl-analysis.git
cd urban-sprawl-analysis
```

> Replace `<your-username>` with the actual GitHub username or organisation hosting this repository.

---

### Step 2 — Upload Data Files to Google Drive

The notebook reads training data from Google Drive. You must mirror the repository's `data/` folder inside your Drive.

**2.1** — Go to [https://drive.google.com](https://drive.google.com) and sign in.

**2.2** — Create the following folder structure exactly as shown (capitalization matters):

```
My Drive/
└── urban_sprawl_analysis/
    ├── data/
    │   ├── Binary_Cairo_Training.csv
    │   └── Categorical_Cairo_Training.csv
    ├── classification/          ← leave empty; Phase 6 exports fill this
    ├── bands_ndvi_bui/          ← leave empty; Phase 6 exports fill this
    └── results/                 ← leave empty; Phase 4 exports fill this
```

**2.3** — Upload `data/Binary_Cairo_Training.csv` and `data/Categorical_Cairo_Training.csv` from the cloned repository into `My Drive/urban_sprawl_analysis/data/`.

> **Verify the path:** Open a Drive file and check that the URL shows the folder path `urban_sprawl_analysis/data/`. The notebook reads from `/content/drive/MyDrive/urban_sprawl_analysis/data/` — any path mismatch will cause a `FileNotFoundError`.

---

### Step 3 — Open the Notebook in Google Colab

**3.1** — Go to [https://colab.research.google.com](https://colab.research.google.com).

**3.2** — Click **File → Upload notebook** and upload:
```
Urban_Sprawl_Analysis_and_classification_and_Visualisation.ipynb
```

Alternatively, you can open it directly from GitHub:
- **File → Open notebook → GitHub tab**
- Paste your repository URL and select the notebook file

**3.3** — Set the Colab runtime to **GPU or High-RAM** for faster processing:
- **Runtime → Change runtime type → Hardware accelerator → GPU**

---

### Step 4 — Enable the Google Earth Engine API

This step is required only once per Google Cloud project.

**4.1** — Go to [https://console.cloud.google.com/apis/library](https://console.cloud.google.com/apis/library).

**4.2** — Search for **"Earth Engine API"** and click **Enable**.

**4.3** — Note your **Project ID** from the top of the Cloud Console (e.g. `nifty-buffer-491115-h9`). You will paste this into the notebook in Phase 2.

---

### Step 5 — Install Dependencies (Phase 1 of the Notebook)

Run the **Phase 1 cells** in order:

**Cell § 1.1 — Install Libraries:**

```python
!pip install earthengine-api geemap --quiet
```

This installs:
- `earthengine-api` — official Python client for GEE server-side operations
- `geemap` — interactive mapping library for rendering GEE images inside Colab

**Cell § 1.2 — Import Libraries:**

```python
import ee, geemap, pandas as pd, matplotlib.pyplot as plt
import numpy as np, seaborn as sns, rasterio
from sklearn.metrics import accuracy_score, confusion_matrix, cohen_kappa_score, f1_score
# ... (full import block in notebook)
```

**Expected output:** Version numbers printed for `earthengine-api`, `geemap`, and `rasterio` with no errors.

> **If you see `ModuleNotFoundError`:** Re-run cell § 1.1, then restart the runtime (**Runtime → Restart runtime**) and re-run § 1.2.

---

### Step 6 — Authenticate Google Earth Engine (Phase 2, § 2.1)

This step must be run at the **start of every Colab session**.

**6.1** — Update the project ID in cell § 2.1:

```python
GEE_PROJECT = 'your-gee-project-id'   # ← paste your Project ID here
```

**6.2** — Run the cell. On **first use**, a browser tab opens asking you to sign in with the Google account linked to your Earth Engine project. Follow these sub-steps:

1. Click the authentication link printed in the cell output
2. Sign in with your Google account in the browser window
3. Click **Allow** to grant Earth Engine access
4. Copy the authorisation code shown in the browser
5. Paste it into the Colab text input box and press Enter

**6.3** — After successful authentication the cell prints:
```
GEE initialised ✓
```

> **If you see `EEException: Earth Engine client is not initialized`:** Verify the project ID is correct and that the Earth Engine API is enabled for that project (Step 4).

> **Session expiry:** GEE credentials are cached for approximately 24 hours. If you open a new Colab session the next day, re-run § 2.1 before any other GEE cell.

---

### Step 7 — Run Phase 2 through Phase 5 (GEE Classification Pipeline)

Phases 2–5 require an active GEE session. Run all cells **top-to-bottom** within each phase.

**Phase 2 — Data Acquisition & Pre-Processing**
- Defines seasonal cloud masking functions for Landsat 5 and 8/9
- Applies QA bit masking for cloud, cloud shadow, cirrus, dilated cloud, and snow/ice
- Applies city-specific seasonal compositing windows to eliminate the 2015 Atlanta cloud anomaly

**Phase 3 — Feature Engineering**
- Computes nine spectral indices (NDVI, NDBI, BUI, MNDWI, NDWI, EVI, BSI, NBI) using sensor-aware band mapping
- Run the interactive visualisation cell to visually confirm index values are physically correct over Cairo 2019

**Phase 4 — Land Cover Classification**
- § 4.1: Trains and applies a 4-class Random Forest (Urban / Vegetation / Water / Bare Soil)
- § 4.2: Trains and applies a binary Random Forest (Urban vs. Non-Urban) across all 4 cities × 6 epochs
- When prompted, the notebook mounts Google Drive to read training CSVs — click **Connect to Google Drive** and grant access

**Phase 5 — Accuracy Evaluation**
- Computes pixel-level accuracy metrics against GHSL P2023A
- Produces confusion matrices and F1-score trend plots for all city-year combinations
- **Target benchmarks:** Overall Accuracy ≥ 88%, Kappa ≥ 0.82, Urban F1 > 0.65

---

### Step 8 — Export GeoTIFFs to Google Drive (Phase 6)

> **Critical:** Phase 7 cannot run until all Phase 6 exports are complete.

**8.1** — Run all cells in Phase 6. Each cell submits GEE export tasks for:
- § 6.1: Binary classification rasters — one `.tif` per city-year (up to 24 files)
- § 6.2: Spectral band rasters — NDVI, NDBI, SWIR — one `.tif` per city-year-band (up to 72 files)

**8.2** — Monitor export progress in the **GEE Code Editor Tasks panel**:
1. Open [https://code.earthengine.google.com](https://code.earthengine.google.com) in a new browser tab
2. Click the **Tasks** tab (top-right)
3. Wait until all tasks show status **COMPLETED** (green check mark)

> Export times vary: a single city-year file typically takes **15–45 minutes** depending on bounding box size and GEE server load. All 24 binary rasters may take **3–8 hours total** to complete if submitted sequentially. You can continue to the dashboard section or close Colab while exports run — they execute on GEE servers independently.

**8.3** — Verify files in Drive:
- Navigate to `My Drive/urban_sprawl_analysis/classification/`
- Confirm files are present, e.g. `paris_2000.tif`, `bangkok_2020.tif`, etc.
- Navigate to `My Drive/urban_sprawl_analysis/bands_ndvi_bui/`
- Confirm spectral files are present, e.g. `atlanta_ndbi_2015.tif`

---

### Step 9 — Run Phase 7: Time-Series Analysis (Local Raster Processing)

Phase 7 runs locally in Colab using exported GeoTIFFs. It does **not** require an active GEE session.

**9.1** — Verify the base path in cell § 7.1 matches your Drive folder structure:

```python
BASE = '/content/drive/MyDrive/urban_sprawl_analysis/classification'
```

> If you used a different folder name in Step 2, update `BASE` here before running any Phase 7 cell.

**9.2** — Mount Google Drive when prompted by the cell:

```python
from google.colab import drive
drive.mount('/content/drive')
```

Click **Connect to Google Drive** and grant access.

**9.3** — Run all Phase 7 cells in order:

| Section | Metric | Output |
|---|---|---|
| § 7.2 | Urban Area Extraction & Anomaly Detection | `areas_df`, `areas_df_clean` |
| § 7.3 | Post-Classification Comparison (Change Maps) | `pcc_df`, change map visualisations |
| § 7.4 | Image Differencing (Raw NDBI) | NDBI difference maps |
| § 7.5 | Change Vector Analysis (3-Band CVA) | `cva_df`, magnitude/direction maps |
| § 7.6 | Urban Growth Rate & CAGR | `growth_df` |
| § 7.7 | Trend Analysis (Linear Regression) | `trend_records`, `trend_df` |
| § 7.8 | Compactness Indices (LSI / LPI / PD) | `compactness_df` |
| § 7.9 | Land Consumption Rate (SDG 11.3.1) | `lcr_df` |
| § 7.10 | Consolidated Visualisations | Interactive Plotly charts |
| § 7.11 | Final Summary Table | `summary_df` |

> **If a raster file is missing:** The cell prints `SKIPPED (file missing or path is None)` and continues — it will not crash. Metrics computed from available files will still be valid.

---

### Step 10 — Launch the Interactive Dashboard (Optional)

The live dashboard visualises all Phase 7 results through a Flask API exposed via Cloudflare Tunnel.

**10.1** — Ensure all Phase 7 cells have been run and all result DataFrames are in memory.

**10.2** — Run the **Dashboard backend cell** (last cell in the notebook). Wait approximately 15 seconds. A public URL is printed:

```
https://workplace-purchases-interstate-philadelphia.trycloudflare.com
```

> This URL is **temporary** — it changes every time the cell is re-run.

**10.3** — Open `Dashboard/dynamic_urban_sprawl.html` in your browser (double-click the file locally or serve it via VS Code Live Server).

**10.4** — Paste the printed Cloudflare URL into the dashboard's **Connect** dialog and click **Connect & Load Dashboard**.

**10.5** — After re-running any analysis cell, click **↻ Refresh Data** in the dashboard to pull updated values.

> **Keep the Colab tab open** while using the dashboard — closing Colab terminates the Flask server and disconnects the dashboard.

---

## Optional: Running Locally Outside Colab

If you prefer to run the notebook on your own machine instead of Colab, follow these additional steps after cloning.

**Install Miniconda** (recommended for environment isolation):
- Download from [https://docs.conda.io/en/latest/miniconda.html](https://docs.conda.io/en/latest/miniconda.html)

**Create and activate a virtual environment:**

```bash
conda create -n urban-sprawl python=3.11 -y
conda activate urban-sprawl
```

**Install all dependencies:**

```bash
pip install -r requirements.txt
```

**Install geospatial system libraries** (required by `rasterio` and `pylandstats`):

- **Ubuntu/Debian:**
  ```bash
  sudo apt-get install -y libgdal-dev gdal-bin libproj-dev
  ```
- **macOS (Homebrew):**
  ```bash
  brew install gdal proj
  ```
- **Windows:** Install [OSGeo4W](https://trac.osgeo.org/osgeo4w/) and add it to your PATH.

**Authenticate GEE locally:**

```bash
earthengine authenticate
```

This opens a browser window — follow the same authorisation flow as Step 6. Credentials are saved locally and do not need to be re-entered.

**Launch Jupyter:**

```bash
jupyter notebook Urban_Sprawl_Analysis_and_classification_and_Visualisation.ipynb
```

> **Note on Google Drive:** When running locally, replace all `/content/drive/MyDrive/...` paths in the notebook with the absolute local path to your data folder, e.g. `/Users/yourname/urban_sprawl_analysis/data/`.

---

## Python Dependencies

All dependencies are listed in `requirements.txt`. Key packages:

```
earthengine-api>=0.1.370
geemap>=0.29.0
numpy>=1.24.0
pandas>=2.0.0
matplotlib>=3.7.0
seaborn>=0.12.0
rasterio>=1.3.0
scipy>=1.10.0
scikit-learn>=1.3.0
plotly>=5.15.0
pylandstats>=2.4.0
flask>=3.0.0
flask-cors>=4.0.0
geopandas>=0.14.0
```

Install all at once:

```bash
pip install -r requirements.txt
```

---

## Data Description

### Training Data

Both CSV files contain pixel-level training samples collected over **Cairo, Egypt** using stratified point sampling in GEE.

| File | Classes | Columns |
|---|---|---|
| `Binary_Cairo_Training.csv` | 0 = Non-Urban, 1 = Urban | `.geo`, `Class`, spectral index values |
| `Categorical_Cairo_Training.csv` | 0 = Bare Soil, 1 = Vegetation, 2 = Water, 3 = Urban | `.geo`, `Class`, spectral index values |

### Exported Results

| File | Description |
|---|---|
| `results/classification_results_all_cities.csv` | RF-predicted urban fraction per city-year |
| `results/truth_values_all_cities.csv` | GHSL ground-truth urban fraction per city-year |
| `classification/{city}_{year}.tif` | Binary classified GeoTIFF at 30 m, EPSG:4326 |
| `bands_ndvi_bui/{city}_{index}_{year}.tif` | Spectral index rasters for CVA (NDVI, NDBI, SWIR) |

---

## Pipeline Architecture

```
Phase 1  →  Phase 2  →  Phase 3  →  Phase 4  →  Phase 5
  Setup      Cloud        Feature      RF           Accuracy
             Masking      Engineering  Classif.     Evaluation
               ↓              ↓            ↓
            Composite     9 Indices    Binary Maps
                                          ↓
                                      Phase 6 Export → Google Drive
                                                            ↓
                                                        Phase 7
                                                     Time Series
                                                      + 8 Metrics
                                                          ↓
                                                      Dashboard
```

---

## Common Issues & Fixes

| Error | Likely Cause | Fix |
|---|---|---|
| `EEException: Earth Engine client is not initialized` | GEE not authenticated or project ID wrong | Re-run § 2.1 with correct project ID |
| `FileNotFoundError: .../Binary_Cairo_Training.csv` | Training CSV not uploaded to Drive or path wrong | Verify Drive folder structure matches Step 2 |
| `ModuleNotFoundError: No module named 'geemap'` | Install cell not run or kernel restarted | Re-run § 1.1 then restart runtime |
| Phase 7 skips all cities | GeoTIFF exports not complete | Wait for all GEE tasks to show COMPLETED |
| Dashboard shows "Connection failed" | Colab tab was closed or Flask cell not running | Re-run the dashboard backend cell |
| `2015 anomaly` still visible after fix | Phase 2 seasonal window not applied before export | Re-export 2015 rasters with updated `get_image()` function |

---

## Execution Order Reference

```
§ 1.1 → § 1.2           Install & import
§ 2.1                    GEE authenticate (required every session)
§ 2.2                    Cloud mask + composite functions
§ 3.1 → § 3.2           Spectral indices + visualisation
§ 4.1.1 → § 4.1.5       Categorical RF classification
§ 4.2.1 → § 4.2.6       Binary RF classification + GHSL validation
§ 5.1 → § 5.2           Accuracy evaluation
§ 6.1 → § 6.2           Export GeoTIFFs → Google Drive
                         ⏳ Wait for GEE exports to complete
§ 7.1 → § 7.11          Time-series metrics (local raster processing)
Dashboard cell           Launch Flask + Cloudflare tunnel
```

---

## Research Objectives

| # | Objective |
|---|---|
| O1 | Sensor-agnostic classification pipeline for Landsat 5 / 7 / 8 / 9 without manual band-list edits |
| O2 | Random Forest achieving Urban F1 > 0.65 across ≥ 3 of the 4 study cities |
| O3 | Diagnose and correct the 2015 Atlanta cloud-mask anomaly |
| O4 | Quantify urban area change 2000–2025 using eight complementary metrics |
| O5 | Compute SDG 11.3.1 LCR per city and interpret against UN sustainability targets |
| O6 | Assess compactness metrics (LSI, LPI, PD) across four climate zones |
| O7 | Document all design decisions for full reproducibility |


---

## License

See [LICENSE](LICENSE) for terms of use.