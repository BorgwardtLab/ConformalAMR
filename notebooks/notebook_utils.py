import warnings

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

### ResAMR-GNN plotting functions


def bootstrap_metrics(y_true, y_pred, bootstraps=1000):
    metrics = defaultdict(list)

    for _ in range(bootstraps):

        # As some datasets are really imbalanced, we use stratified bootstrapping to get CIs
        pos_indices = np.random.choice(
            np.array(range(len(y_true)))[y_true == 1],
            len(y_true[y_true == 1]),
            replace=True,
        )
        neg_indices = np.random.choice(
            np.array(range(len(y_true)))[y_true == 0],
            len(y_true[y_true == 0]),
            replace=True,
        )
        indices = np.concatenate([pos_indices, neg_indices])

        metrics["average_precision"].append(
            average_precision_score(
                y_true[indices],
                y_pred[indices],
            )
        )
        metrics["roc_auc"].append(
            roc_auc_score(
                y_true[indices],
                y_pred[indices],
            )
        )

    # ROC AUC
    metrics["roc_auc_low"], metrics["roc_auc_high"] = np.nanpercentile(
        metrics["roc_auc"], [2.5, 97.5]
    )
    metrics["roc_auc"] = np.nanmedian(metrics["roc_auc"])

    # AUPRC
    metrics["average_precision_low"], metrics["average_precision_high"] = (
        np.nanpercentile(metrics["average_precision"], [2.5, 97.5])
    )
    metrics["average_precision"] = np.nanmedian(metrics["average_precision"])

    return metrics


def get_metrics(data, species="all", drug="all", threshold=0.5, bootstraps=1000):
    data.response = data.response.astype(int)

    if species != "all":
        data = data[data["species"] == species]
    if drug != "all":
        data = data[data["drug"] == drug]

    try:
        metrics = {
            "average_precision": (
                average_precision_score(data["response"], data["Predictions"])
                if bootstraps == 1
                else []
            ),
            "roc_auc": (
                roc_auc_score(data["response"], data["Predictions"])
                if bootstraps == 1
                else []
            ),
        }
        if bootstraps > 1:
            metrics = bootstrap_metrics(
                data["response"].values,
                data["Predictions"].values,
                bootstraps=bootstraps,
            )

    except NotImplementedError:  # (IndexError, ValueError):
        metrics = {
            "average_precision": np.nan,
            "roc_auc": np.nan,
        }

    return metrics


def get_metrics_baselines(data, bootstraps=1000):

    try:
        with open(data, "r") as f:
            data = json.load(f)

        y_true = np.array(data["y_test"])
        y_pred = np.array(data["y_score"])[:, 1]

        metrics = {
            "average_precision": average_precision_score(y_true, y_pred),
            "roc_auc": roc_auc_score(y_true, y_pred),
        }
        if bootstraps > 1:
            metrics = bootstrap_metrics(y_true, y_pred, bootstraps=bootstraps)

    except (FileNotFoundError, ValueError):
        metrics = {
            "average_precision": np.nan,
            "roc_auc": np.nan,
        }
        if bootstraps > 1:
            metrics = {
                "roc_auc": np.nan,
                "average_precision": np.nan,
                "roc_auc_low": np.nan,
                "average_precision_low": np.nan,
                "roc_auc_high": np.nan,
                "average_precision_high": np.nan,
            }

    return metrics


def get_curves(data, species=None, drug=None, bootstrap=1):
    data.response = data.response.astype(int)

    if species != "all":
        data = data[data["species"] == species]
    if drug != "all":
        data = data[data["drug"] == drug]

    fpr, tpr, _ = roc_curve(data["response"], data["Predictions"])
    precision, recall, _ = precision_recall_curve(data["response"], data["Predictions"])

    return fpr, tpr, precision, recall


def get_curves_baselines(data):

    with open(data, "r") as f:
        data = json.load(f)

    y_true = np.array(data["y_test"])
    y_pred = np.array(data["y_score"])[:, 1]

    fpr, tpr, _ = roc_curve(y_true, y_pred)
    precision, recall, _ = precision_recall_curve(y_true, y_pred)

    return fpr, tpr, precision, recall


def plot_curves(
    results_dict, species="all", drugs="all", model=None, bootstrap=1, save=False
):

    results_list, run_names = [], []
    for key, data in results_dict.items():
        results_list.append(data)
        run_names.append(" ".join(key.split("_")))

    if isinstance(drugs, str):
        drugs = [drugs] * len(results_list)
    else:
        results_list *= len(drugs)
        run_names = drugs

    fig, ax = plt.subplots(1, 2, figsize=(15, 7), sharey=True)

    for j, data in enumerate(results_list):

        if isinstance(data, pd.DataFrame):
            fpr, tpr, precision, recall = get_curves(data, species, drugs[j], bootstrap)
            data_metrics = get_metrics(data, species, drugs[j], bootstraps=bootstrap)

        elif isinstance(data, str):
            data = data.replace("__MODEL__", model)
            data = data.replace("__SPECIES__", species.replace(" ", "_"))
            data = data.replace("__DRUG__", drugs[j].replace(" ", "_"))
            fpr, tpr, precision, recall = get_curves_baselines(data)
            data_metrics = get_metrics_baselines(data)

        ax[0].plot(
            fpr,
            tpr,
            label="AUC = {:.2f} | {}".format(data_metrics["roc_auc"], run_names[j]),
        )
        ax[0].set_xlabel("False Positive Rate")
        ax[0].set_ylabel("True Positive Rate")
        ax[0].set_title("ROC Curve | Species: {}".format(species))

        ax[1].plot(
            recall,
            precision,
            label="AUC = {:.2f} | {}".format(
                data_metrics["average_precision"], run_names[j]
            ),
        )
        ax[1].set_xlabel("Recall")
        ax[1].set_ylabel("Precision")

        ax[1].set_title("Precision-Recall Curve | Species: {}".format(species))

    # Add random line
    ax[0].plot([0, 1], [0, 1], linestyle="--", color="darkgrey")
    if len(set(drugs)) == 1:
        ax[1].plot(
            [0, 1], [precision[0], precision[0]], linestyle="--", color="darkgrey"
        )

    ax[0].legend()
    ax[1].legend()

    # Add dashed grid
    ax[0].grid(True, linestyle="--", alpha=0.6)
    ax[1].grid(True, linestyle="--", alpha=0.6)

    # Remove top and right spines
    ax[0].spines["top"].set_visible(False)
    ax[0].spines["right"].set_visible(False)
    ax[1].spines["top"].set_visible(False)
    ax[1].spines["right"].set_visible(False)

    plt.tight_layout()
    if save:
        plt.savefig(save, dpi=400)
    plt.show()


### Conformal prefiction plotting functions


def get_fig_dim(width=900, fraction=0.33, aspect_ratio=None):
    """Set figure dimensions to avoid scaling in LaTeX.
    Parameters
    ----------
    width: float
            Document textwidth or columnwidth in pts
    fraction: float, optional
            Fraction of the width which you wish the figure to occupy
    Returns
    -------
    fig_dim: tuple
            Dimensions of figure in inches
    """
    # Width of figure (in pts)
    fig_width_pt = width * fraction

    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Golden ratio to set aesthetic figure height
    if aspect_ratio is None:
        aspect_ratio = (1 + 5**0.5) / 2
        # aspect_ratio =  2.5

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = fig_width_in / aspect_ratio

    fig_dim = (fig_width_in, fig_height_in)

    return fig_dim


def marginal_group_figures(df_results_cross, folder_path):
    sns.set_context("paper")

    df_results_cross.sort_values(by=["Model", "alpha", "tau", "split"], inplace=True)

    w, h = get_fig_dim(width=469, fraction=1.0)
    fig, axes = plt.subplots(1, 3, figsize=(1.1 * w, h / 1.5))
    ax = sns.lineplot(
        x="alpha",
        y="FDR",
        hue="Model",
        style="tau",
        data=df_results_cross,
        legend=False,
        palette="husl",
        ax=axes[0],
    )
    sns.despine(ax=ax)
    ax.grid(linestyle="--", alpha=0.2)
    ax.set_xlabel(r"$\alpha$")

    ax = sns.lineplot(
        x="alpha",
        y="FNR",
        hue="Model",
        style="tau",
        data=df_results_cross,
        legend=False,
        palette="husl",
        ax=axes[1],
    )
    sns.despine(ax=ax)
    ax.grid(linestyle="--", alpha=0.2)
    ax.set_xlabel(r"$\alpha$")

    ax = sns.lineplot(
        x="alpha",
        y="Coverage",
        hue="Model",
        style="tau",
        data=df_results_cross,
        legend="full",
        palette="husl",
        ax=axes[2],
    )

    sns.despine(ax=ax)
    ax.grid(linestyle="--", alpha=0.2)
    ax.set_xlabel(r"$\alpha$")

    legend = ax.legend(prop={"size": 9}, frameon=False)
    for vpack in legend._legend_handle_box.get_children():
        for hpack in vpack.get_children():
            draw_area, text_area = hpack.get_children()
            for collection in text_area.get_children():
                if collection.get_text() == "tau":
                    text_area.set_text(r"$\tau$")

    sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))

    plt.tight_layout()
    plt.savefig(folder_path + "/baseline_all.pdf", format="pdf", bbox_inches="tight")
    plt.show()


def hyperparam_study_tau_figure(df_hyperparam_results, folder_path):
    warnings.filterwarnings("ignore")
    sns.set_context("paper")
    df_hyperparam_results.sort_values(
        by=["Model", "alpha", "tau", "split"], inplace=True
    )

    for model in df_hyperparam_results.Model.unique():
        w, h = get_fig_dim(width=487, fraction=0.5)
        fig, ax = plt.subplots(figsize=(w, h))

        df_plot = (
            df_hyperparam_results[df_hyperparam_results["Model"] == model]
            .groupby(["Model", "alpha", "tau", "fold"])
            .mean()
            .reset_index()
        )
        mean_auc = df_plot.groupby(["Model", "alpha", "tau"]).mean().reset_index()

        df_plot["AUC"] = df_plot[["alpha", "tau"]].apply(
            lambda a: mean_auc[(mean_auc["alpha"] == a[0]) & (mean_auc["tau"] == a[1])][
                "AUC"
            ].values,
            axis=1,
        )
        df_plot["hue"] = df_plot[["tau", "AUC"]].apply(
            (lambda m: str(m[0]) + ", " + str(np.round(m[1], 3))), axis=1
        )
        ax = sns.lineplot(
            x="alpha",
            y="FDR",
            hue="hue",
            data=df_plot,
            legend="full",
            palette="crest_r",
        )

        sns.despine(ax=ax)
        ax.legend(title=r"$\tau$," + " [AUC]", prop={"size": 9}, frameon=False)
        ax.set_xlabel(r"$\alpha$")
        ax.grid(linestyle="--", alpha=0.2)
        sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
        plt.savefig(
            folder_path + "/hyparam_tau_" + model + ".pdf",
            format="pdf",
            bbox_inches="tight",
        )
        plt.show()


def hyperparam_study_aggfunc_figure(df_hyperparam_results, folder_path):
    warnings.filterwarnings("ignore")
    sns.set_context("paper")
    df_hyperparam_results.sort_values(
        by=["Model", "alpha", "tau", "split"], inplace=True
    )

    for tau in df_hyperparam_results.tau.unique():
        w, h = get_fig_dim(width=487, fraction=0.5)
        fig, ax = plt.subplots(figsize=(w, h))

        df_plot = (
            df_hyperparam_results[df_hyperparam_results["tau"] == tau]
            .groupby(["Model", "alpha", "tau", "fold"])
            .mean()
            .reset_index()
        )
        mean_auc = df_plot.groupby(["Model", "alpha", "tau"]).mean().reset_index()
        df_plot["AUC"] = df_plot[["alpha", "Model"]].apply(
            lambda a: mean_auc[
                (mean_auc["alpha"] == a[0]) & (mean_auc["Model"] == a[1])
            ]["AUC"].values,
            axis=1,
        )
        df_plot["hue"] = df_plot[["Model", "AUC"]].apply(
            (lambda m: m[0].replace("_", " ") + ", " + str(np.round(m[1], 3))), axis=1
        )
        ax = sns.lineplot(
            x="alpha", y="FDR", hue="hue", data=df_plot, legend="full", palette="husl"
        )

        sns.despine(ax=ax)
        ax.grid(linestyle="--", alpha=0.2)
        ax.set_xlabel(r"$\alpha$")

        ax.legend(
            title="Agg. function," + " [AUC]",
            ncol=2,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.5),
            prop={"size": 8},
            frameon=False,
        )
        plt.savefig(
            folder_path + "/hyparam_agg_" + str(tau) + ".pdf",
            format="pdf",
            bbox_inches="tight",
        )
        plt.show()


def individual_coverage_figures(df_results_per_drug, folder_path):
    sns.set_context("paper")
    tau_list = [0.0, 1.0]

    drug_types = {
        "Imipenem": "Carbapenems",
        "Meropenem": "Carbapenems",
        "Ertapenem": "Carbapenems",
        "Ciprofloxacin": "Fluoroquinolones",
        "Levofloxacin": "Fluoroquinolones",
        "Cefepime": "Cephalosporins",
        "Ceftriaxone": "Cephalosporins",
        "Ceftazidime": "Cephalosporins",
        "Piperacillin-Tazobactam": "Penicillins",
        "Amoxicillin-Clavulanic acid": "Penicillins",
    }

    df_results_per_drug["Type"] = df_results_per_drug["Drug"].map(drug_types)
    df_plot = df_results_per_drug[df_results_per_drug["tau"].isin(tau_list)][
        [
            "Type",
            "Model",
            "Drug",
            "alpha",
            "tau",
            "split",
            "FDR",
            "Coverage",
            "Conditional Coverage",
        ]
    ]
    df_plot["Models"] = df_plot[["Model", "tau"]].apply(
        (lambda m: m[0] + r", $\tau$=" + str(m[1])), axis=1
    )

    for drug_type in df_results_per_drug["Type"].unique():
        w, h = get_fig_dim(width=469, fraction=1.0)
        fig, axes = plt.subplots(1, 2, figsize=(1.1 * w, h / 1.5))
        ax = sns.lineplot(
            x="alpha",
            y="FDR",
            style="Drug",
            hue="Models",
            data=df_plot[df_plot["Type"] == drug_type],
            palette="husl",
            markers=True,
            dashes=False,
            ax=axes[0],
            legend=False,
        )
        sns.despine(ax=ax)
        ax.set_xlabel(r"$\alpha$")
        ax.grid(linestyle="--", alpha=0.2)
        ax = sns.lineplot(
            x="alpha",
            y="Coverage",
            style="Drug",
            hue="Models",
            data=df_plot[df_plot["Type"] == drug_type],
            palette="husl",
            markers=True,
            dashes=False,
            ax=axes[1],
        )

        sns.despine(ax=ax)
        ax.set_xlabel(r"$\alpha$")
        ax.grid(linestyle="--", alpha=0.2)
        legend = ax.legend(prop={"size": 9}, frameon=False)

        for vpack in legend._legend_handle_box.get_children():
            for hpack in vpack.get_children():
                draw_area, text_area = hpack.get_children()
                for collection in text_area.get_children():
                    if collection.get_text() == "Models":
                        text_area.set_text("Model")
                    if collection.get_text() == "Piperacillin-Tazobactam":
                        text_area.set_text("P.-Tazobactam")
                    if collection.get_text() == "Amoxicillin-Clavulanic acid":
                        text_area.set_text("A.-Clavulanic acid")
        sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))

        plt.tight_layout()
        plt.savefig(
            folder_path + "/per_drug_" + drug_type + ".pdf",
            format="pdf",
            bbox_inches="tight",
        )

        plt.show()
