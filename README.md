# Heterogeneity-Aware Fusion Learning for Lung Adenocarcinoma (LUAD) 
## Project Overview
This project implements a **multimodal deep learning pipeline** that is designed to predict 5-year mortality in Lung Adenocarcinoma (LUAD) patients. By integrating four distinct data modalities, this model is able to overcome the limitations of unimodal analysis and addresses the challenge of **data heterogeneity** in medical AI.
### The Modalities
1.  **Clinical Data:** Tabular demographic and exposure data.
2.  **Genomics (SNV):** Somatic mutation data (binary matrix).
3.  **Transcriptomics (RNA-Seq):** Gene expression profiles (log-transformed).
4.  **Histopathology (WSI):** Whole-Slide Images processed via ResNet50.

## Data Access
**Note:** The raw data for this project is hosted on a Google Cloud Storage (GCS) Bucket due to its size (>100GB).

### Public Data Links
You do not need authentication to access these files. The notebook is configured to pull them automatically, but you can inspect them here:

* **Clinical Data:**  clinical.tsv (https://storage.googleapis.com/nguyend9_final_project/clinical.tsv)
* **Follow-up Data:** follow_up.tsv (https://storage.googleapis.com/nguyend9_final_project/follow_up.tsv)
* **Pathology Details:** pathology_detail.tsv (https://storage.googleapis.com/nguyend9_final_project/pathology_detail.tsv)
* **Exposure Data:** exposure.tsv (https://storage.googleapis.com/nguyend9_final_project/exposure.tsv)
* **Family History:** family_history.tsv (https://storage.googleapis.com/nguyend9_final_project/family_history.tsv)

## Architecture & Methodology
### 1. Data Pipeline
* **Atomic Checkpointing:** The pipeline saves progress after every 5 patients to prevent data loss during long feature extraction runs.
* **Batched Inference:** Image patches are processed in batches of 64 on the GPU.
* **Imbalance Handling:** The target label (5-year mortality) was engineered to create a balanced 70:30 class distribution (Negative/Positive).

### 2. Feature Extraction
* **Imaging:** `OpenSlide` tiles images into 256x256 patches. A pre-trained **ResNet50** (truncated at the FC layer) extracts a 2048-dimensional feature vector for every patch.
* **Omics:** Dimensionality reduction via variance thresholding (Top 100 genes).

### 3. Fusion Model
The model employs a **Late Fusion** strategy (SimpleFusion), concatenating feature vectors from all modalities into a shared latent space before passing them through a multi-layer perceptron for binary classification.

## Running the project/code
Open `NguyenDavid_FinalProject.ipynb`
1.  **Preprocessing:** The script will automatically download clinical data and begin extracting image features.
2.  **Training:** The model will train for 100 epochs (with early stopping).
3.  **Evaluation:** ROC Curves, Confusion Matrices, and UMAP plots will be generated in the outputs.

It took me 12+ hours to run the code on Google Colab after purchasing the Pro version. I had to buy multiple hours of units as Google Vertex AI was unable to run and kept giving me an error. This code would require modifications in order to run using Vertex AI.
