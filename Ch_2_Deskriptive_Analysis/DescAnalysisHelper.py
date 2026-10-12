import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import matplotlib.ticker as mtick
import numpy as np
import ast
TYPES = ["SFH", "TH", "MFH", "AB"]
COLOR_DICT = {"SFH": "#2f6db5", "TH": "#e69f00", "MFH": "#2a9d8f", "AB": "#c44e52"}

def plot_manifest_static_building(manifest):
    types = TYPES
    fig, axes = plt.subplots(1, 2, figsize=(10, 7))
    ax = axes.flat[0]
    val_counts = manifest["buildingType"].value_counts()
    ax.bar(types, val_counts.values, color=[COLOR_DICT[i] for i in types],alpha=1)
    for i, v in enumerate(val_counts.values):
        ax.text(i, v + val_counts.max() * 0.015, f"{v:,}\n{v / len(manifest):.0%}", ha="center", fontsize=8)
    ax.set_title("Building type", loc="left", fontweight="bold")
    ax.set_ylabel("buildings")
    ax.grid(axis="y", alpha=0.7)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax = axes.flat[1]
    values = manifest.groupby(["buildingAgeBin", "buildingType"]).size().unstack(fill_value=0)
    x = np.arange(len(values))
    b = np.zeros(len(values))
    for t in types:
        ax.bar(x, values[t], bottom=b, width=0.6, color=COLOR_DICT[t],alpha=1 ,edgecolor="white", linewidth=0.8)
        b += values[t].values
    ax.set_xticks(x, values.index.astype(str))
    ax.tick_params(axis="x", labelrotation=45)
    for i, tot in enumerate(b):
        ax.text(i, tot + b.max() * 0.015, f"{int(tot):,}", ha="center", fontsize=8)
    ax.set_title("Building Age Bin", loc="left", fontweight="bold")
    ax.set_ylabel("buildings")
    ax.grid(axis="y", alpha=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    fig.legend(handles=[Patch(color=COLOR_DICT[t], label=t) for t in types], title="Building type",
               loc="upper right", ncol=4, frameon=False)

    fig.tight_layout(rect=(0, 0, 1, 0.94))


def plot_manifest_static_weather(manifest, cols=["year_type", "future", "climateRegion"]):
    types = TYPES
    fig, axes = plt.subplots(1, 3, figsize=(14, 6))
    col_dict = {"year_type": "Year Type", "future": "Future", "climateRegion": "Climate Region"}
    for ax, c in zip(axes.flat, cols):
        values = manifest.groupby([c, "buildingType"]).size().unstack(fill_value=0)
        x = np.arange(len(values))
        b = np.zeros(len(values))
        for t in types:
            ax.bar(x, values[t], bottom=b, width=0.6, color=COLOR_DICT[t], edgecolor="white", linewidth=0.8)
            b += values[t].values
        ax.set_xticks(x, values.index.astype(str))
        ax.tick_params(axis="x", labelrotation=45)

        for i, tot in enumerate(b):
            ax.text(i, tot + b.max() * 0.015, f"{int(tot):,}", ha="center", fontsize=8)
            ax.set_title(col_dict[c], loc="left", fontweight="bold")
            ax.set_ylabel("buildings")
            ax.grid(axis="y", alpha=0.7)
            ax.spines[["top", "right"]].set_visible(False)
            ax.set_axisbelow(True)

    fig.legend(handles=[Patch(color=COLOR_DICT[t], label=t) for t in types], title="Building type",
               loc="upper right", ncol=4, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.94))

def plot_manifest_numeric(manifest,cols=["a_ref","n_apartments","n_persons_sum"]):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    col_dict = {"a_ref": "Total Living Space in m²", "n_apartments": "Number of Apartments per Building", "n_persons_sum": "Total number of persons per building","n_appliances": "Total number of unique Appliance Types per Building" }
    units = {"a_ref": "buildings", "n_apartments": "buildings", "n_persons_sum": "buildings","n_appliances": "runs" }
    for ax, col in zip(axes.flat,cols):
        vals = manifest[col]
        if col == "a_ref":
            bins = np.linspace(vals.min(), vals.max(), 41)
        else:
            bins = np.arange(vals.min(), vals.max() + 2) - 0.5
        ax.hist([manifest.loc[manifest["buildingType"]==bldg, col] for bldg in TYPES], bins=bins,stacked=True, color=[COLOR_DICT[bldg] for bldg in TYPES],edgecolor="white", linewidth=0.4)
        ax.axvline(vals.median(), color="black", ls="--", lw=1.2)
        ax.text(0.5, 0.95, f"n = {len(vals):,} {units[col]}\nmedian {vals.median():.3g}\nmean {vals.mean():.3g}\n"
                            f"range {vals.min():.3g} – {vals.max():.3g}",
                transform=ax.transAxes, ha="right", va="top", fontsize=9)
        ax.set_title(col_dict[col], loc="left", fontweight="bold")
        ax.set_ylabel(units[col])
        ax.grid(axis="y", alpha=0.7)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_axisbelow(True)

    fig.legend(handles=[Patch(color=COLOR_DICT[t], label=t) for t in TYPES], title="Building type",
               loc="upper right", ncol=4, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.94))

def plot_appliance_ownership(manifest):
    app = manifest["appliances_unique"].map(lambda s: ast.literal_eval(s) if isinstance(s, str) else s)
    app_df = pd.DataFrame({"buildingType": manifest["buildingType"], "appliance":app}).explode("appliance")
    owned = app_df.groupby("appliance").size().sort_values()
    owned_rel = owned/len(manifest)
    labels = owned_rel.index.str.replace("_", " ")
    fig, ax = plt.subplots(figsize=(5,7))
    ax.barh(labels, owned_rel.values,color="#2f6db5", height=0.65)
    for i, v in enumerate(owned_rel.values):
        ax.text(v + 0.01, i, f"{v:.0%}", va="center", fontsize=9)

    ax.xaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.set_xlim(0, 1.12)
    ax.set_xticks(np.arange(0, 1.01, 0.2))
    ax.tick_params(axis="y", length=0)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.grid(axis="x", color="0.85", linewidth=0.7)
    ax.set_axisbelow(True)
    ax.set_title(f"Appliance ownership", loc="left", fontweight="bold")
    ax.set_xlabel("Share of buildings")
    fig.tight_layout()


def boxplot_specific_sh(ts, y="sh_sum_yearly_per_a_ref", ylabel="space heating in kWh/(m²*a)", logy=False):
    ages = sorted(ts["buildingAgeBin"].unique())
    fig, axes = plt.subplots(1, 4, figsize=(18, 5), sharey=True)
    for ax, t in zip(axes.flat, TYPES):
        sub = ts[ts.buildingType == t]
        groups = [sub.loc[sub.buildingAgeBin == a, y].values for a in ages]
        bp = ax.boxplot(groups, positions=range(len(ages)), widths=0.6, patch_artist=True, whis=1.5,
                        medianprops=dict(color="#1f2933", lw=1.3),
                        boxprops=dict(facecolor=COLOR_DICT[t], alpha=0.55, edgecolor=COLOR_DICT[t]),
                        whiskerprops=dict(color=COLOR_DICT[t]), capprops=dict(color=COLOR_DICT[t]),
                        flierprops=dict(marker=".", markersize=3, markeredgecolor="#6b7785", alpha=0.5))
        ax.set_xticks(range(len(ages)), [a.replace("DE.", "") for a in ages])
        ax.set_xlabel("Building Age Bin")
        ax.set_title(f"{t} (n = {len(sub):,} runs)", loc="left", fontweight="bold")
        ax.grid(axis="y", color="#e4e8ed"); ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        if logy: ax.set_yscale("log")
    axes[0].set_ylabel(ylabel)
    fig.suptitle("Specific space heating demand by building type and age class",
                 x=0.01, ha="left", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))

def boxplot_specific_energy(ts, y="sh_sum_yearly_per_a_ref", ylabel="space heating in kWh/(m²*a)",title="Domestic Hot Water demand by building type and age class", logy=False, param="buildingAgeBin"):
    fig, axes = plt.subplots(1, 4, figsize=(18, 5), sharey=True)
    for ax, t in zip(axes.flat, TYPES):
        sub = ts.loc[ts["buildingType"] == t]
        params = sorted(sub[param].dropna().unique())
        sub = ts[ts.buildingType == t]
        groups = [sub.loc[sub[param] == a, y].values for a in params]
        ax.boxplot(groups, positions=range(len(params)), widths=0.6, patch_artist=True, whis=1.5,
                        medianprops=dict(color="#1f2933", lw=1.3),
                        boxprops=dict(facecolor=COLOR_DICT[t], alpha=0.55, edgecolor=COLOR_DICT[t]),
                        whiskerprops=dict(color=COLOR_DICT[t]), capprops=dict(color=COLOR_DICT[t]),
                        flierprops=dict(marker=".", markersize=3, markeredgecolor="#6b7785", alpha=0.5))
        ax.set_xticks(range(len(params)))
        ax.set_xticklabels(params)
        ax.tick_params("x", rotation=45, rotation_mode="xtick")
        ax.set_title(f"{t} (n = {len(sub):,} runs)", loc="left", fontweight="bold")
        ax.grid(axis="y", color="#e4e8ed"); ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        if logy: ax.set_yscale("log")
    axes[0].set_ylabel(ylabel)
    fig.suptitle(title,
                 x=0.01, ha="left", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))