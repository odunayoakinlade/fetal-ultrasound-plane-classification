# AI-Assisted Fetal Ultrasound Plane Classification for Acquisition Support in Low-Resource Settings

**Course:** 04-652 Artificial Intelligence System Design, Carnegie Mellon University Africa
**Author:** Odunayo Wuraola Akinlade
**Assignment 2:** Data Preprocessing (Phase 1: Strategy · Phase 2: Implementation)

---

## Problem Statement

Fetal ultrasound is a core part of antenatal care. The WHO recommends one ultrasound scan before 24 weeks of pregnancy. In much of Sub-Saharan Africa, though, there are few ultrasound specialists. Trained nurses and midwives increasingly perform point-of-care obstetric scans and often have no expert available to review an image right away. Models trained in other clinical settings also tend to generalise poorly to African data, because the ultrasound machines, acquisition practices and patient populations differ.

**Objective:** build a lightweight model that takes a single fetal ultrasound image and classifies it into one of four standard anatomical planes: **abdomen, brain, femur or thorax**. The model also returns a confidence score, and low-confidence predictions are flagged for human review or reacquisition. The system supports acquisition. It does not certify clinical image quality or replace trained personnel.

**SDG alignment:** SDG 3 (Good Health & Well-being; Targets 3.1, 3.2) and SDG 10 (Reduced Inequalities).

## Dataset Overview

### Target dataset (primary, preprocessed in this assignment)

| Item | Value |
| --- | --- |
| Name | *Maternal fetal ultrasound planes from low-resource imaging settings in five African countries* (Sendra-Balcells et al., 2023) |
| Source | <https://doi.org/10.5281/zenodo.7540447> |
| Sample size (**N**) | **450** images (one metadata row per image) |
| Metadata columns (**P**) | **5**: `Patient_num`, `Plane`, `Train`, `Center`, `Filename` |
| Image features | 2-D B-mode PNGs at 5 native resolutions (400–600 px wide × 400–500 px tall), with 350 grayscale and 100 RGB. Images will be resized to a common input size, so the model input dimensionality is set in the Phase 1 strategy. |
| Target variable (**Y**) | `Plane` ∈ {Fetal abdomen, Fetal brain, Fetal femur, Fetal thorax} |
| Grouping unit | 127 unique patient–centre combinations (`Center` + `Patient_num`; patient numbers repeat across countries) |
| Countries | Algeria (100), Egypt (100), Malawi (100), Ghana (75), Uganda (75) |
| Class balance | Abdomen 125 · Brain 125 · Femur 125 · Thorax 75 (Ghana and Uganda have no thorax images) |
| Missing values | 0 in metadata; all 450 records match an image file |

### Source dataset (pretraining, used in later project stages)

*FETAL_PLANES_DB* (Burgos-Artizzu et al., 2020): 12,400 images from 1,792 patients at two hospitals in Spain. Source: <https://doi.org/10.5281/zenodo.3904279>.

## Task Type

**Multi-Class Classification** (4 classes, single label per image).

## Repository Structure

```
fetal-ultrasound-plane-classification/
├── .gitignore                 # Excludes data binaries, caches, virtual envs, model weights
├── requirements.txt           # Pinned package dependencies
├── README.md                  # This guide
├── data/
│   ├── raw/                   # Original, unaltered dataset (not tracked in git)
│   └── processed/             # Cleaned, encoded and scaled features (not tracked in git)
├── phase1_strategy/
│   └── Phase1_Strategy_Report.pdf
├── phase2_implementation/
│   ├── notebooks/
│   │   └── pipeline_execution.ipynb
│   └── Phase2_Final_Report.pdf
├── src/
│   ├── __init__.py
│   ├── transformers.py        # Custom scikit-learn transformers
│   └── pipeline_builder.py    # ColumnTransformer / Pipeline definitions
└── figures/                   # 300-DPI diagnostic plots (fig1_1 … fig5_1)
```

## Reproducing the Environment

```bash
git clone https://github.com/<your-username>/fetal-ultrasound-plane-classification.git
cd fetal-ultrasound-plane-classification
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Getting the data

The dataset is not committed to this repository because of its size and licensing. To set it up:

1. Download the archive from <https://doi.org/10.5281/zenodo.7540447>.
2. Extract it into `data/raw/` so that the folder contains:

```
data/raw/
├── African_planes_database.csv
├── Algeria/  ├── Egypt/  ├── Ghana/  ├── Malawi/  └── Uganda/
```

## Pipeline Summary

*To be completed in Phase 2.*

## Deliverables

| Phase | Deliverable | Status |
| --- | --- | --- |
| 1 | `phase1_strategy/Phase1_Strategy_Report.pdf` | In progress |
| 2 | `phase2_implementation/notebooks/pipeline_execution.ipynb` | Not started |
| 2 | `phase2_implementation/Phase2_Final_Report.pdf` | Not started |
