import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import auc
from sklearn.model_selection import KFold

from conformal_amr.conformal_models import conformal_models

relevant_drugs_per_species = {
    "Escherichia coli": {
        "highly relevant": [
            "Amoxicillin-Clavulanic acid",
            "Piperacillin-Tazobactam",
            "Ceftriaxone",
            "Ceftazidime",
            "Cefepime",
            "Ertapenem",
            "Ciprofloxacin",
            "Levofloxacin",
        ],
    },
    "Klebsiella pneumoniae": {
        "highly relevant": [
            "Amoxicillin-Clavulanic acid",
            "Piperacillin-Tazobactam",
            "Ceftriaxone",
            "Ceftazidime",
            "Cefepime",
            "Ertapenem",
            "Imipenem",
            "Meropenem",
            "Ciprofloxacin",
            "Levofloxacin",
        ],
    },
}


def hamming_loss(y_true, y_pred):
    temp = 0
    for i in range(y_true.shape[0]):
        temp += (~np.isnan(y_true[i])).sum() - np.count_nonzero(y_true[i] == y_pred[i])
    return temp / (y_true.size - np.count_nonzero(np.isnan(y_true)))


def get_driams_data(part, species):
    driams_scores = pd.read_csv(
        "../../results/Stratified_by_species/Mole_BERT/ConformalAMR/Conformal_DRIAMS-"
        + part
        + "_spectra_results/finetuned_models/finetuned_results_test_extended.csv"
    )
    driams_scores = driams_scores[driams_scores["species"] == species]

    drug_list = [
        drug
        for drug in relevant_drugs_per_species[species]["highly relevant"]
        if drug in set(driams_scores["drug"].unique())
    ]
    driams_scores = driams_scores[driams_scores["drug"].isin(drug_list)]

    driams_scores_wide = driams_scores.pivot(
        index=["species", "sample_id"], columns="drug", values="Predictions"
    )
    driams_response_wide = driams_scores.pivot(
        index=["species", "sample_id"], columns="drug", values="response"
    )
    drug_list = driams_response_wide.columns
    driams_scores = driams_scores_wide.to_numpy()
    driams_labels = driams_response_wide.to_numpy()

    return (
        driams_scores,
        driams_labels,
        driams_scores_wide.index.get_level_values(0),
        drug_list,
    )


def calibration_split(cal_ratio, driams_scores, driams_labels):

    n = int(cal_ratio * driams_scores.shape[0])
    idx = np.array([1] * n + [0] * (driams_scores.shape[0] - n)) > 0
    np.random.shuffle(idx)
    cal_clf_scores, val_clf_scores = driams_scores[idx, :], driams_scores[~idx, :]
    cal_labels, val_labels = driams_labels[idx, :], driams_labels[~idx, :]

    return cal_clf_scores, val_clf_scores, cal_labels, val_labels


def run_hyperparam_experiment(
    species, alpha_list, driamsA_drugs, driamsA_scores, driamsA_labels, max_iter=10
):

    warnings.filterwarnings("ignore")

    results = []
    results_per_drug = []
    hyperparam_results = []

    for fold in range(max_iter):
        print("fold: ", fold)
        # test-calibration split
        random_noise = (0.2) * np.random.random_sample() - 0.1
        fold_clf_scores, test_clf_scores, fold_labels, test_labels = calibration_split(
            0.75 + random_noise, driamsA_scores, driamsA_labels
        )
        print(
            "test ratio:",
            1 - (0.75 + random_noise),
            ", test size: ",
            test_clf_scores.shape[0],
        )

        # calibration-validation split
        kf = KFold(n_splits=3, shuffle=True)
        fdr_aucs = []
        for i, (train_index, val_index) in enumerate(kf.split(fold_clf_scores)):
            cal_clf_scores, val_clf_scores = (
                fold_clf_scores[train_index, :],
                fold_clf_scores[val_index, :],
            )
            cal_labels, val_labels = (
                fold_labels[train_index, :],
                fold_labels[val_index, :],
            )

            model_list = ["max", "mean", "filtered_mean", "softmax"]
            tau_hyperparam_list = [0.0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0]

            for tau in tau_hyperparam_list:

                config = {
                    "cal_size": cal_labels.shape[0],
                    "cal_labels": cal_labels,
                    "cal_clf_scores": cal_clf_scores,
                    "tau": tau,
                    "agg_type": "max",
                    "drug_list": driamsA_drugs,
                    "alpha": 0.1,
                }
                MaxKGCP = conformal_models.KGConformalPredictor(config)
                val_prediction_sets_MaxKGCP = MaxKGCP.predict_all_alpha(
                    val_clf_scores, alpha_list=alpha_list
                )

                config = {
                    "cal_size": cal_labels.shape[0],
                    "cal_labels": cal_labels,
                    "cal_clf_scores": cal_clf_scores,
                    "tau": tau,
                    "agg_type": "mean",
                    "drug_list": driamsA_drugs,
                    "alpha": 0.1,
                }
                MeanKGCP = conformal_models.KGConformalPredictor(config)
                val_prediction_sets_MeanKGCP = MeanKGCP.predict_all_alpha(
                    val_clf_scores, alpha_list=alpha_list
                )

                config = {
                    "cal_size": cal_labels.shape[0],
                    "cal_labels": cal_labels,
                    "cal_clf_scores": cal_clf_scores,
                    "tau": tau,
                    "agg_type": "filtered_mean",
                    "drug_list": driamsA_drugs,
                    "alpha": 0.1,
                }
                MeanFKGCP = conformal_models.KGConformalPredictor(config)
                val_prediction_sets_MeanFKGCP = MeanFKGCP.predict_all_alpha(
                    val_clf_scores, alpha_list=alpha_list
                )

                config = {
                    "cal_size": cal_labels.shape[0],
                    "cal_labels": cal_labels,
                    "cal_clf_scores": cal_clf_scores,
                    "tau": tau,
                    "softmax_factor": 1,
                    "agg_type": "softmax",
                    "drug_list": driamsA_drugs,
                    "alpha": 0.1,
                }
                SoftmaxKGCP = conformal_models.KGConformalPredictor(config)
                val_prediction_sets_SoftmaxKGCP = SoftmaxKGCP.predict_all_alpha(
                    val_clf_scores, alpha_list=alpha_list
                )

                for name, prediction_set in {
                    "max": val_prediction_sets_MaxKGCP,
                    "filtered_mean": val_prediction_sets_MeanFKGCP,
                    "mean": val_prediction_sets_MeanKGCP,
                    "softmax": val_prediction_sets_SoftmaxKGCP,
                }.items():

                    # FDR
                    fdr_vec = [
                        np.mean(
                            1
                            - np.nan_to_num(
                                (
                                    np.nansum(
                                        val_labels * prediction_set[alpha], axis=1
                                    )
                                    / np.nansum(
                                        (~np.isnan(val_labels)).astype(int)
                                        * prediction_set[alpha],
                                        axis=1,
                                    )
                                ),
                                nan=1.0,
                            )
                        )
                        for alpha in alpha_list
                    ]
                    auc_value = auc(alpha_list, fdr_vec)

                    fdr_aucs += [
                        {
                            "Model": name,
                            "tau": tau,
                            "split": i,
                            "fold": fold,
                            "AUC": auc_value,
                        }
                    ]

                    hyperparam_results += [
                        {
                            "Model": name,
                            "tau": tau,
                            "split": i,
                            "fold": fold,
                            "FDR": fdr_vec[index],
                            "alpha": alpha,
                            "AUC": auc_value,
                        }
                        for index, alpha in enumerate(alpha_list)
                    ]

        df_fdr_aucs = (
            pd.DataFrame(fdr_aucs)
            .groupby(["Model", "tau"])
            .agg("mean")
            .reset_index()[["Model", "tau", "AUC"]]
        )
        # print(df_fdr_aucs)
        best_param = df_fdr_aucs.loc[df_fdr_aucs.idxmin().loc["AUC"]]
        print(best_param)
        # print(best_param.loc["Model"])

        config = {
            "cal_size": fold_labels.shape[0],
            "cal_labels": fold_labels,
            "cal_clf_scores": fold_clf_scores,
            "tau": best_param.loc["tau"],
            "softmax_factor": 1,
            "agg_type": best_param.loc["Model"],
            "drug_list": driamsA_drugs,
            "alpha": 0.1,
        }
        BestKGCP = conformal_models.KGConformalPredictor(config)
        test_prediction_sets_BestKGCP = BestKGCP.predict_all_alpha(
            test_clf_scores, alpha_list=alpha_list
        )

        config = {
            "cal_size": fold_labels.shape[0],
            "cal_labels": fold_labels,
            "cal_clf_scores": fold_clf_scores,
            "tau": 0.0,
            "softmax_factor": 1,
            "agg_type": best_param.loc["Model"],
            "drug_list": driamsA_drugs,
            "alpha": 0.1,
        }
        Baseline = conformal_models.KGConformalPredictor(config)
        test_prediction_sets_Baseline = Baseline.predict_all_alpha(
            test_clf_scores, alpha_list=alpha_list
        )

        df_labels = pd.DataFrame(test_labels, columns=driamsA_drugs)

        for alpha in alpha_list:
            for model, prediction_set in (
                {
                    ("Baseline", 0.0): test_prediction_sets_Baseline[alpha],
                    (
                        best_param.loc["Model"],
                        best_param.loc["tau"],
                    ): test_prediction_sets_BestKGCP[alpha],
                }
            ).items():

                # FDR
                fdr = np.mean(
                    1
                    - np.nan_to_num(
                        (
                            np.nansum(test_labels * prediction_set, axis=1)
                            / np.nansum(
                                (~np.isnan(test_labels)).astype(int) * prediction_set,
                                axis=1,
                            )
                        ),
                        nan=1.0,
                    )
                )

                # conformal coverage
                coverage = (np.any((test_labels - prediction_set) == 1, axis=1)).mean()

                # hamming loss
                hl = hamming_loss(test_labels, prediction_set)

                # fnr
                fnr = np.mean(
                    1
                    - np.nan_to_num(
                        (
                            np.nansum(test_labels * prediction_set, axis=1)
                            / np.nansum(
                                (~np.isnan(test_labels)).astype(int) * test_labels,
                                axis=1,
                            )
                        ),
                        nan=1.0,
                    )
                )

                df_sets = pd.DataFrame(prediction_set, columns=driamsA_drugs)
                results.append(
                    {
                        "Model": model[0],
                        "alpha": alpha,
                        "tau": model[1],
                        "split": fold,
                        "FDR": fdr,
                        "FNR": fnr,
                        "Coverage": coverage,
                        "Hamming Loss": hl,
                    }
                )

                results_per_drug += [
                    {
                        "Model": model[0],
                        "alpha": alpha,
                        "tau": model[1],
                        "split": fold,
                        "Drug": drug,
                        "Coverage": 1 - (df_labels[drug] <= df_sets[drug]).mean(),
                        "FDR": (np.nansum((1 - df_labels[drug]) * df_sets[drug]))
                        / np.nansum(
                            (df_labels[drug].notna()).astype(int) * df_sets[drug]
                        ),
                        "Conditional Coverage": 1
                        - (df_sets[df_labels[drug] == 1][drug] == 1).mean(),
                    }
                    for drug in driamsA_drugs
                ]

    df_results_cross = pd.DataFrame(results)
    df_results_per_drug = pd.DataFrame(results_per_drug)
    df_hyperparam_results = pd.DataFrame(hyperparam_results)

    # Create output folder if it does not exist
    os.makedirs("../results/Conformal_Evaluation", exist_ok=True)

    df_results_cross.to_csv(
        "../results/Conformal_Evaluation/df_results_cross_" + species + ".csv"
    )
    df_results_per_drug.to_csv(
        "../results/Conformal_Evaluation/df_results_per_drug_" + species + ".csv"
    )

    df_hyperparam_results.to_csv(
        "../results/Conformal_Evaluation/df_hyperparam_results_" + species + ".csv"
    )


def main():

    np.random.seed(873926459)
    alpha_list = [0.01, 0.05, 0.08, 0.10, 0.12, 0.15, 0.18]

    print("Klebsiella pneumoniae")
    driamsA_scores, driamsA_labels, driamsA_species, driamsA_drugs = get_driams_data(
        "A", "Klebsiella pneumoniae"
    )
    run_hyperparam_experiment(
        "Klebsiella pneumoniae",
        alpha_list,
        driamsA_drugs,
        driamsA_scores,
        driamsA_labels,
        max_iter=30,
    )
    print("Escherichia coli")
    driamsA_scores, driamsA_labels, driamsA_species, driamsA_drugs = get_driams_data(
        "A", "Escherichia coli"
    )
    run_hyperparam_experiment(
        "Escherichia coli",
        alpha_list,
        driamsA_drugs,
        driamsA_scores,
        driamsA_labels,
        max_iter=30,
    )


if __name__ == "__main__":
    main()
