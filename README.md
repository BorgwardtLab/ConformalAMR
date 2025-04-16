# ConformalAMR

This repository contains the code used to conduct experiments presented in the paper _Detecting antimicrobial resistance through MALDI-TOF mass spectrometry with statistical guarantees using conformal prediction_, by Nina Corvelo Benz, Lucas Miranda, Dexiong Chen, Janko Sattler, and Karsten Borgwardt.

### Abstract

Antimicrobial resistance (AMR) is a global health challenge, complicating the treatment of
bacterial infections and leading to higher patient morbidity and mortality. Rapid and reliable identification of resistant pathogens is crucial to guide early and effective therapeutic interventions. However, traditional culture-based methods are time-consuming, highlighting the need for faster predictive approaches. Machine learning models trained on MALDI-TOF mass spectrometry data, readily collected
in most clinics for fast species identification, offer promise but face limitations in clinical applicability, particularly due to their lack of comprehensive, statistically valid uncertainty estimates. Here, we introduce a novel AMR prediction framework that addresses this gap with a novel knowledge graphenhanced conformal predictor. Conformal prediction (CP) constructs prediction sets with statistical
coverage guarantees, ensuring that bacterial resistance to a certain antibiotic is flagged with a specified
error rate. 

![Conformal framework](images/pipeline.png "Conformal prediction pipeline.")

Our proposed conformal predictor constructs improved prediction sets over standard CP approaches by using a knowledge graph capturing the interdependencies in antibiotic resistance patterns.

![Conformal performance](images/conformal.png "Our knowledge-graph-enhanced conformal predictor leads to a lower FDR than non-enhanced baselines for a fixed confidence score.")

In addition, we introduce a novel classifier framework that improves upon previous multimodal models by incorporating multigraph-based antibiotic representations using state-of-the-art self-supervised
methods. Besides increasing resistance detection for most tested species-drug combinations, the presented architecture, termed ResMLP-GNN, overcomes the limitations of previous efforts and supports
multi-drug antibiotics that are highly relevant in clinical practice. We successfully evaluated our approach on a set of highly-relevant antibiotics, commonly used in clinics to treat infections with Klebsiella pneumoniae and Escherichia coli

![ROC and PR curves](images/curves.png "ResMLP-GNN reaches SOTA performance when predicting antimicrobial resistance from MALDI-TOF and drug structure.")

### Installation

We provide `pyproject.toml` and `poetry.lock` files for reproducible installation of the environment used in our experiments using [poetry](https://python-poetry.org/). We recommend doing this within a conda/mamba isolated environment, running `Python^=3.11`. To install the environment, just run the following commands (assuming you have poetry up and running):

```bash

mamba create -n ConformalAMR python=3.11
poetry install

```

This should install both the required dependencies, and the conformal_amr package as such.

### Training base ResAMR-GNN models

To train the base ResAMR-GNN models, you can run the following command:

```bash

poetry run python scripts/train_ResAMR_classifier.py --driams_dataset A --driams_long_table data/DRIAMS_combined_long_table_multidrug.csv --drugs_df data/DRIAMS_Mole-BERT_drug_embeddings.csv --spectra_matrix path/to/spectra/DRIAMS-A/spectra_binned_6000_all_multidrug.npy --output_folder results

```

Where `A` can be replaced by `B` or `C` or `D` to train on the respective DRIAMS datasets, and the spectra matrix is generated using the `notebooks/Process_DRIAMS_data.ipynb.py` notebook, upon downloading the DRIAMS dataset.

### Finetuning ResMLP-GNN models

To finetune a base ResMLP-GNN model on a fixed species-drug combination, you can run the following command:

```bash

poetry run python scripts/finetune_ResAMR_classifier.py --pretrained_checkpoints_folder /path/to/pretrained/base/models --driams_dataset A --driams_long_table data/DRIAMS_combined_long_table_multidrug.csv --drugs_df data/DRIAMS_Mole-BERT_drug_embeddings.csv --spectra_matrix path/to/spectra/DRIAMS-A/spectra_binned_6000_all_multidrug.npy --splits_file path/to/splits/file --n_epochs 150 --output_folder results --species_drug_combination 'Escherichia coli_Ceftriaxone' --freeze_drug_emb
    
```

Where `A` can be replaced by `B` or `C` or `D` to train on the respective DRIAMS datasets, the pretrained checkpoints and the splits files are generated after training the base models, and 'Escherichia coli_Ceftriaxone' can be replaced by any other species-drug combination of interest.

### Evaluating the conformal predictor described in the paper

To evaluate knowledge-graph enhanced conformal predictor with the hyperparameters described in the paper, you can simply run:

```bash

poetry run python scripts/evaluate_conformal.py

```
