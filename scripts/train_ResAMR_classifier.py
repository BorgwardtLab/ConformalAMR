import itertools
import json
import os
import sys
from argparse import ArgumentParser
from os.path import exists, join

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import shap
import torch
import torch.nn as nn
import wandb
from lightning.pytorch.loggers import WandbLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torch.utils.data import DataLoader
from tqdm import tqdm

from conformal_amr.data_splits.data_utils import DataSplitter
from conformal_amr.experiments.pl_experiment import Classifier_Experiment
from conformal_amr.models.classifier import Residual_AMR_Classifier
from conformal_amr.models.data_loaders import (
    DrugResistanceDataset_Embeddings,
    DrugResistanceDataset_Fingerprints,
    SampleEmbDataset,
)

# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


def main(args):
    config = vars(args)
    seed = args.seed
    # Setup output folders to save results

    if args.sweep is not None:
        random_id = f"_{wandb.util.generate_id()}"
        args.experiment_name += random_id

    output_folder = join(
        args.output_folder, args.experiment_group, args.experiment_name, str(args.seed)
    )
    if not exists(output_folder):
        os.makedirs(output_folder, exist_ok=True)

    results_folder = join(
        args.output_folder, args.experiment_group, args.experiment_name + "_results"
    )
    if not exists(results_folder):
        os.makedirs(results_folder, exist_ok=True)

    experiment_folder = join(
        args.output_folder, args.experiment_group, args.experiment_name
    )

    if exists(join(results_folder, f"test_metrics_{args.seed}.json")):
        sys.exit(0)
    if not exists(experiment_folder):
        os.makedirs(experiment_folder, exist_ok=True)

    # Read data
    driams_long_table = pd.read_csv(args.driams_long_table)

    # If multiple matrices are provided, load and concatenate all of them
    spectra_matrices = []
    for spectra_path in args.spectra_matrix.split(","):
        spectra_matrices.append(np.load(spectra_path).astype(float))
    spectra_matrix = np.vstack(spectra_matrices)

    drugs_df = pd.read_csv(args.drugs_df, index_col=0)
    driams_long_table = driams_long_table[
        driams_long_table["drug"].isin(drugs_df.index)
    ]

    # Instantate data split
    dsplit = DataSplitter(driams_long_table, dataset=args.driams_dataset)
    samples_list = sorted(dsplit.long_table["sample_id"].unique())

    # Split selection for the different experiments.
    if args.splits_file is not None:
        test_samples = pd.read_csv(args.splits_file, index_col=0)["sample_id"].values
        trainval_df = dsplit.long_table.loc[
            ~dsplit.long_table["sample_id"].isin(test_samples)
        ].reset_index(drop=True)
        test_df = dsplit.long_table.loc[
            dsplit.long_table["sample_id"].isin(test_samples)
        ].reset_index(drop=True)

        # Split the trainval set into train and validation
        trainval_dsplit = DataSplitter(trainval_df, dataset=args.driams_dataset)
        train_df, val_df = trainval_dsplit.sample_id_train_test_split(
            test_size=0.05, random_state=args.seed
        )

    elif args.split_type == "random":
        train_df, val_df, test_df = dsplit.random_train_val_test_split(
            val_size=0.1, test_size=0.4, random_state=args.seed
        )
    elif args.split_type == "spectra":

        if args.test_species is not None:
            args.test_species = args.test_species.split(",")

        trainval_df, test_df = dsplit.sample_id_train_test_split(
            test_size=0.4, select_species=args.test_species, random_state=args.seed
        )

        # Split the trainval set into train and validation
        trainval_dsplit = DataSplitter(trainval_df, dataset=args.driams_dataset)
        train_df, val_df = trainval_dsplit.sample_id_train_test_split(
            test_size=0.05,
            select_species=args.test_species,
            reverse_sp_selection=True,
            random_state=args.seed,
        )

    elif args.split_type == "drug_species_zero_shot":
        trainval_df, test_df = dsplit.combination_train_test_split(
            dsplit.long_table, test_size=0.4, random_state=args.seed
        )
        train_df, val_df = dsplit.baseline_train_test_split(
            trainval_df, test_size=0.05, random_state=args.seed
        )
    elif args.split_type == "drugs_zero_shot":
        drugs_list = sorted(dsplit.long_table["drug"].unique())
        if args.seed >= len(drugs_list):
            print("Drug index out of bound, exiting..\n\n")
            sys.exit(0)
        target_drug = drugs_list[args.seed]
        # target_drug = args.drug_name
        test_df, trainval_df = dsplit.drug_zero_shot_split(drug=target_drug)
        train_df, val_df = dsplit.baseline_train_test_split(
            trainval_df, test_size=0.2, random_state=args.seed
        )

    test_df.to_csv(join(output_folder, "test_set.csv"), index=False)

    if args.drug_emb_type == "fingerprint":
        train_dset = DrugResistanceDataset_Fingerprints(
            train_df,
            spectra_matrix,
            drugs_df,
            samples_list,
            fingerprint_class=config["fingerprint_class"],
        )
        val_dset = DrugResistanceDataset_Fingerprints(
            val_df,
            spectra_matrix,
            drugs_df,
            samples_list,
            fingerprint_class=config["fingerprint_class"],
        )
        test_dset = DrugResistanceDataset_Fingerprints(
            test_df,
            spectra_matrix,
            drugs_df,
            samples_list,
            fingerprint_class=config["fingerprint_class"],
        )
    elif args.drug_emb_type == "vae_embedding" or args.drug_emb_type == "gnn_embedding":
        train_dset = DrugResistanceDataset_Embeddings(
            train_df, spectra_matrix, drugs_df, samples_list
        )
        val_dset = DrugResistanceDataset_Embeddings(
            val_df, spectra_matrix, drugs_df, samples_list
        )
        test_dset = DrugResistanceDataset_Embeddings(
            test_df, spectra_matrix, drugs_df, samples_list
        )

    sorted_species = sorted(dsplit.long_table["species"].unique())
    idx2species = {i: s for i, s in enumerate(sorted_species)}
    species2idx = {s: i for i, s in idx2species.items()}

    config["n_unique_species"] = len(idx2species)
    del config["seed"]
    # Save configuration
    if not exists(join(experiment_folder, "config.json")):
        with open(join(experiment_folder, "config.json"), "w") as f:
            json.dump(config, f)
    if not exists(join(results_folder, "config.json")):
        with open(join(results_folder, "config.json"), "w") as f:
            json.dump(config, f)

    train_loader = DataLoader(
        train_dset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        val_dset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    test_loader = DataLoader(
        test_dset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    # Instantiate logger
    run_tags = [
        "ConformalAMR",
        "Base_model_training",
        args.experiment_group,
        args.experiment_name,
        f"Split={args.split_type}",
        f"DRIAMS-{args.driams_dataset}",
        f"Drug_emb_type={args.drug_emb_type}",
        f"Drug_emb_size={args.drug_embedding_dim}",
        f"Droput={args.dropout}",
    ]
    wandb_logger = WandbLogger(
        project="ConformalAMR",
        entity="lucas_miranda",
        tags=run_tags,
    )

    # Instantiate model and pytorch lightning experiment
    model = Residual_AMR_Classifier(config)
    experiment = Classifier_Experiment(config, model)

    # Save summary of the model architecture
    if not exists(join(experiment_folder, "architecture.txt")):
        with open(join(experiment_folder, "architecture.txt"), "w") as f:
            f.write(model.__repr__())
    if not exists(join(results_folder, "architecture.txt")):
        with open(join(results_folder, "architecture.txt"), "w") as f:
            f.write(model.__repr__())

    # Setup training callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(output_folder, "checkpoints"),
        monitor="val_loss",
        save_top_k=5,
        save_last=True,
        filename="gst-{epoch:03d}-{val_loss:.4f}",
    )
    early_stopping_callback = EarlyStopping(
        monitor="val_loss", mode="min", patience=args.patience
    )
    callbacks = [checkpoint_callback, early_stopping_callback]

    # Train model
    print("Training..")
    trainer = pl.Trainer(
        devices=1,
        accelerator="auto",
        default_root_dir=output_folder,
        max_epochs=args.n_epochs,
        callbacks=callbacks,
        logger=wandb_logger,
        log_every_n_steps=3,
        # limit_train_batches=6, limit_val_batches=4, limit_test_batches=4,
    )
    trainer.fit(experiment, train_dataloaders=train_loader, val_dataloaders=val_loader)
    print("Training complete!")

    # Test model
    print("Testing..")
    test_results = trainer.test(ckpt_path="best", dataloaders=test_loader)
    with open(join(results_folder, "test_metrics_{}.json".format(seed)), "w") as f:
        json.dump(test_results[0], f, indent=2)

    background_loader = DataLoader(test_dset, batch_size=100, shuffle=True)
    background = next(iter(background_loader))
    X_background = torch.cat(background[:3], dim=1)

    train_fi_loader = DataLoader(train_dset, batch_size=len(train_dset), shuffle=False)
    test_fi_loader = DataLoader(test_dset, batch_size=len(test_dset), shuffle=False)
    train_fi_batch, fi_batch = next(iter(train_fi_loader)), next(iter(test_fi_loader))
    train_X_fi_batch, X_fi_batch = torch.cat(train_fi_batch[:3], dim=1), torch.cat(
        fi_batch[:3], dim=1
    )

    # Setting the model to eval mode
    experiment.model.eval()

    train_df["Predictions"] = (
        torch.sigmoid(experiment.model(train_fi_batch)).detach().cpu().numpy()
    )
    train_df.to_csv(join(results_folder, f"train_set_seed{seed}.csv"), index=False)

    test_df["Predictions"] = experiment.test_predictions
    test_df.to_csv(join(results_folder, f"test_set_seed{seed}.csv"), index=False)

    if args.eval_importance:
        print("Evaluating feature importance")

        class ShapWrapper(nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model

            def forward(self, X):
                species_idx = X[:, 0:1]
                x_spectrum = X[:, 1:6001]
                dr_tensor = X[:, 6001:]
                response = []
                dataset = []
                batch = [species_idx, x_spectrum, dr_tensor, response, dataset]
                return experiment.model(batch)

        shap_wrapper = ShapWrapper(experiment.model)
        explainer = shap.DeepExplainer(shap_wrapper, X_background)
        shap_values = explainer.shap_values(X_fi_batch)
        column_names = (
            ["Species"]
            + [f"Spectrum_{i}" for i in range(6000)]
            + [f"Fprint_{i}" for i in range(1024)]
        )
        np.save(join(results_folder, f"shap_values_seed{seed}.npy"), shap_values)
        if not exists(join(output_folder, "shap_values_columns.json")):
            with open(join(output_folder, "shap_values_columns.json"), "w") as f:
                json.dump(column_names, f, indent=2)

        shap_values_df = pd.DataFrame(shap_values, columns=column_names)
        shap_values_df.to_csv(join(output_folder, "shap_values.csv"), index=False)

    print("Testing complete")


if __name__ == "__main__":

    parser = ArgumentParser()

    parser.add_argument("--experiment_name", type=str, default="Conformal")
    parser.add_argument("--experiment_group", type=str, default="ConformalAMR")
    parser.add_argument("--output_folder", type=str, default="outputs")
    parser.add_argument(
        "--split_type",
        type=str,
        default="spectra",
        choices=["random", "spectra", "drug_species_zero_shot", "drugs_zero_shot"],
    )
    parser.add_argument("--test_species", type=str, default=None)

    parser.add_argument("--eval_importance", action="store_true")

    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument(
        "--driams_dataset",
        type=str,
        default="A",
        help="DRIAMS dataset to use. If multiple datasets are used, provide a list of datasets separated by commas.",
    )

    # Add external data and conserve splits on different runs
    parser.add_argument("--splits_file", type=str)

    parser.add_argument("--driams_long_table", type=str)
    parser.add_argument(
        "--spectra_matrix",
        type=str,
        help="Path to the spectra matrix. If training on multiple datasets, provide a list of paths separated by commas.",
    )
    parser.add_argument("--drugs_df", type=str)

    parser.add_argument("--conv_out_size", type=int, default=512)
    parser.add_argument("--sample_embedding_dim", type=int, default=512)
    parser.add_argument("--drug_embedding_dim", type=int, default=128)

    parser.add_argument(
        "--drug_emb_type",
        type=str,
        default="fingerprint",
        choices=["fingerprint", "vae_embedding", "gnn_embedding"],
    )
    parser.add_argument(
        "--fingerprint_class",
        type=str,
        default="morgan_1024",
        choices=["all", "MACCS", "morgan_512", "morgan_1024", "pubchem", "none"],
    )
    parser.add_argument("--fingerprint_size", type=int, default=1024)

    parser.add_argument("--n_hidden_layers", type=int, default=7)

    parser.add_argument("--n_epochs", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--patience", type=int, default=70)
    parser.add_argument("--learning_rate", type=float, default=0.0025)
    parser.add_argument("--weight_decay", type=float, default=1e-05)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--sweep", type=str, default=None)

    args = parser.parse_args()
    args.num_workers = 1
    args.species_embedding_dim = 128

    args.experiment_name = (
        args.experiment_name + f"_DRIAMS-{args.driams_dataset}_{args.split_type}"
    )

    main(args)
