# AI-Assisted Fetal Ultrasound Plane Classification for Acquisition Support in Low-Resource Settings

**Author:** Odunayo Wuraola Akinlade

---

## Problem Statement

Fetal ultrasound is a core part of antenatal care. The WHO recommends one ultrasound scan before 24 weeks of pregnancy. In much of Sub-Saharan Africa, though, there are few ultrasound specialists. Trained nurses and midwives increasingly perform point-of-care obstetric scans and often have no expert available to review an image right away. Models trained in other clinical settings also tend to generalise poorly to images acquired with different equipment and settings, because the ultrasound machines, acquisition practices and image quality differ.

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
| Feature count (**P**) | 5 metadata columns (Patient_num, Plane, Train, Center, Filename), plus the ultrasound image for each row |
| Target variable (**Y**) | Plane: Fetal abdomen, Fetal brain, Fetal femur or Fetal thorax |

### Source dataset (pretraining)

| Item | Value |
| --- | --- |
| Name | *FETAL_PLANES_DB: Common maternal-fetal ultrasound images* (Burgos-Artizzu et al., 2020) |
| Source | <https://doi.org/10.5281/zenodo.3904279> |
| Sample size (**N**) | 12,400 images from 1,792 patients |
| Feature count (**P**) | 7 metadata columns (Image_name, Patient_num, Plane, Brain_plane, Operator, US_Machine, Train), plus the ultrasound image for each row |
| Target variable (**Y**) | Plane, limited to the same four planes as the target dataset |

## Task Type

**Multi-Class Classification**: each image gets exactly one of 4 labels.

## Repository Structure

```
fetal-ultrasound-plane-classification/
├── .gitignore                 # Excludes data binaries, caches, virtual envs, model weights
├── requirements.txt           # Package dependencies
├── README.md                  # This guide
├── data/
│   ├── raw/                   # Original, unaltered datasets (not tracked in git)
│   │   ├── african_planes/    #   Target dataset (fine-tuning)
│   │   └── fetal_planes_db/   #   Source dataset (pretraining)
│   └── processed/             # Cleaned tables, splits and standardised images (not tracked in git)
├── phase1_strategy/
│   └── Phase1_Strategy_Report.pdf     # Data audit and preprocessing strategy
├── phase2_implementation/
│   ├── notebooks/
│   │   └── pipeline_execution.ipynb   # End-to-end pipeline and diagnostics
│   └── Phase2_Final_Report.pdf        # Implementation report and decision log
├── src/
│   ├── __init__.py
│   ├── transformers.py        # Image functions and custom scikit-learn transformers
│   └── pipeline_builder.py    # Pipeline and ColumnTransformer definitions
└── figures/                   # 300-DPI diagnostic plots
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

1. Download the African dataset from <https://doi.org/10.5281/zenodo.7540447>, extract it into `data/raw/african_planes/`, and rename its CSV file to `african_planes.csv`.
2. Download FETAL_PLANES_DB from <https://doi.org/10.5281/zenodo.3904279> (about 2 GB), extract it into `data/raw/fetal_planes_db/`, and rename its CSV file to `fetal_planes_data.csv`.

The resulting layout should be:

```
data/raw/
├── african_planes/
│   ├── african_planes.csv
│   └── Algeria/  Egypt/  Ghana/  Malawi/  Uganda/
└── fetal_planes_db/
    ├── fetal_planes_data.csv
    └── Images/
```

Then open `phase2_implementation/notebooks/pipeline_execution.ipynb` and run all cells. A full run takes several minutes, mostly to standardise the ~6,000 source images, and downloads the ResNet18 weights (about 45 MB) the first time.

## Pipeline Summary

| Stage | What it does | Result |
| --- | --- | --- |
| Duplicate removal | MD5 and 64-bit perceptual hashes; images within a Hamming distance of 4 are treated as copies, and conflicting-label copies are dropped | Target 450 → 396 images; source 6,561 → 6,059 |
| Split | `StratifiedGroupKFold` by patient, stratified by plane | Target 321 development / 75 test; source 5,220 training / 839 validation; no patient in two partitions |
| Outlier check | 1.5 × IQR on brightness inside the fan, per site, fitted on development data | 9 of 321 flagged; kept, with per-image percentile clipping as treatment |
| Image pipeline | Grayscale → fan mask → crop → clip to 1st–99th percentile → pad and resize | 224 × 224 images; 3 × 224 × 224 model input with ImageNet scaling |
| Measurements | Eight image measurements; Spearman \|ρ\| ≥ 0.85 and VIF > 10 screening, then scaling, in one `ColumnTransformer` | 8 → 5 measurements, all VIF < 3 |
| Embeddings | ResNet18 embeddings, standardised, then PCA | 512 → 139 components for 95% of variance |
| Checks | Leakage and format checks | 12 of 12 pass |

Everything that learns from data is fitted on the development (or training) partition only. The image steps learn nothing, so the same pipeline is applied unchanged to every partition.

## Documentation

- [Data audit and preprocessing strategy](phase1_strategy/Phase1_Strategy_Report.pdf)
- [Implementation report and decision log](phase2_implementation/Phase2_Final_Report.pdf)
