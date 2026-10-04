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

The project uses two datasets with the same anatomical plane labels. The model is pretrained on the larger source dataset and then fine-tuned on the African target dataset.

### Target dataset (fine-tuning)

| Item | Value |
| --- | --- |
| Name | *Maternal fetal ultrasound planes from low-resource imaging settings in five African countries* (Sendra-Balcells et al., 2023) |
| Source | <https://doi.org/10.5281/zenodo.7540447> |
| Sample size (**N**) | 450 images |
| Feature count (**P**) | 5 metadata columns (`Patient_num`, `Plane`, `Train`, `Center`, `Filename`), plus the ultrasound image for each row |
| Target variable (**Y**) | `Plane`: Fetal abdomen, Fetal brain, Fetal femur or Fetal thorax |

### Source dataset (pretraining)

| Item | Value |
| --- | --- |
| Name | *FETAL_PLANES_DB: Common maternal-fetal ultrasound images* (Burgos-Artizzu et al., 2020) |
| Source | <https://doi.org/10.5281/zenodo.3904279> |
| Sample size (**N**) | 12,400 images from 1,792 patients |
| Feature count (**P**) | 7 metadata columns (`Image_name`, `Patient_num`, `Plane`, `Brain_plane`, `Operator`, `US_Machine`, `Train`), plus the ultrasound image for each row |
| Target variable (**Y**) | `Plane`, limited to the same four planes as the target dataset |

## Task Type

**Multi-Class Classification**: each image gets exactly one of 4 labels.

## Repository Structure

```
fetal-ultrasound-plane-classification/
├── .gitignore                 # Excludes data binaries, caches, virtual envs, model weights
├── requirements.txt           # Pinned package dependencies
├── README.md                  # This guide
├── data/
│   ├── raw/                   # Original, unaltered datasets (not tracked in git)
│   │   ├── african_planes/    #   Target dataset (fine-tuning)
│   │   └── fetal_planes_db/   #   Source dataset (pretraining)
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
git clone https://github.com/odunayoakinlade/fetal-ultrasound-plane-classification.git
cd fetal-ultrasound-plane-classification
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Getting the data

The datasets are not committed to this repository. Together they are about 2 GB, which is over GitHub's size limits, and each one is distributed under its own Zenodo licence. To set them up:

1. Download the African dataset from <https://doi.org/10.5281/zenodo.7540447> and extract it into `data/raw/african_planes/`.
2. Download FETAL_PLANES_DB from <https://doi.org/10.5281/zenodo.3904279> (about 2 GB) and extract it into `data/raw/fetal_planes_db/`.

The resulting layout should be:

```
data/raw/
├── african_planes/
│   ├── African_planes_database.csv
│   └── Algeria/  Egypt/  Ghana/  Malawi/  Uganda/
└── fetal_planes_db/
    ├── FETAL_PLANES_DB_data.csv
    └── Images/
```

## Pipeline Summary

*To be completed in Phase 2.*

## Deliverables

| Phase | Deliverable | Status |
| --- | --- | --- |
| 1 | `phase1_strategy/Phase1_Strategy_Report.pdf` | In progress |
| 2 | `phase2_implementation/notebooks/pipeline_execution.ipynb` | Not started |
| 2 | `phase2_implementation/Phase2_Final_Report.pdf` | Not started |
