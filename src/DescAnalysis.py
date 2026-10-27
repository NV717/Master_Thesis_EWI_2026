import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import matplotlib.ticker as mtick
from matplotlib.lines import Line2D
from collections import Counter
import numpy as np
import ast
import seaborn as sns
from Ch_1_Data_Preparation.DataPreparationHelper import *
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


def plot_parameter_scatter_cont(data, en_type, params, figsize=(15,13)):
    fig, axes = plt.subplots(len(en_type),len(params), figsize=figsize, squeeze=False)
    for i ,(en_row, en_label) in enumerate(en_type.items()):
        for j ,(param_row, param_label) in enumerate(params.items()):
            ax = axes[i,j]
            #x_vals = data[param_row]
            for t in TYPES:
                y_vals = data[data["buildingType"]==t][en_row]
                x_vals = data[data["buildingType"]==t][param_row]
                ax.scatter(x_vals,y_vals,s=3, alpha=0.15,color=COLOR_DICT[t], edgecolor=None)
            x = data[param_row].to_numpy()
            y = data[en_row].to_numpy()


            mask = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)

            log_x = np.log10(x[mask])
            log_y = np.log10(y[mask])

            corr = np.corrcoef(log_x, log_y)[0, 1]

            p = np.polynomial.polynomial.Polynomial.fit(np.log10(data[param_row]), np.log10(data[en_row]), deg=1)
            b,m = p.convert().coef

            corr = np.corrcoef(np.log10(data[param_row]), np.log10(data[en_row]))[0, 1]

            x_slope = np.array([data[param_row].min(), data[param_row].max()])
            y_slope = 10** p(np.log10(x_slope))

            ax.plot(x_slope, y_slope, color="#1f2933", lw=1.2, ls="--")


            ax.set_xscale("log")
            ax.set_yscale("log")

            ax.text(
                0.04, 0.95,
                f"r = {corr:.2f}",
                transform=ax.transAxes,
                va="top",
                fontsize=9
            )
            ax.grid(True, which="both", color="#e4e8ed", lw=0.5); ax.set_axisbelow(True)
            ax.spines[["top", "right"]].set_visible(False)
            if j == 0: ax.set_ylabel(en_label)
            if i == 2: ax.set_xlabel(param_label)

    fig.legend([Line2D([], [], marker="o", ls="", color=COLOR_DICT[t], label=t) for t in TYPES], TYPES,
               title="Building type", loc="upper right", ncol=4, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.95))

def scatter_specific_energy_by_building_type(
    ts, y="sh_sum_yearly_per_a_ref", x="temperature_mean",
    ylabel="space heating in kWh/(m²*a)",
    title="Space heating demand by building type and age class",
    logy=False
):
    fig, axes = plt.subplots(1, 4, figsize=(18, 5), sharey=True)

    age_bins = sorted(ts["buildingAgeBin"].dropna().unique())
    colors = plt.cm.viridis(np.linspace(0, 1, len(age_bins)))
    age_colors = dict(zip(age_bins, colors))
    for ax, t in zip(axes, TYPES):
        sub = ts.loc[ts["buildingType"] == t]

        for age in age_bins:
            age_sub = sub.loc[sub["buildingAgeBin"] == age]
            ax.scatter(age_sub[x],age_sub[y],s=3,alpha=0.25,color=age_colors[age],edgecolor="none")

        valid = sub[[x, y]].dropna()
        valid = valid[(valid[x] > 0) & (valid[y] > 0)]

        if len(valid) >= 2:
            log_x = np.log10(valid[x])
            log_y = np.log10(valid[y])

            p = np.polynomial.polynomial.Polynomial.fit(log_x, log_y, deg=1)

            b, m = p.convert().coef
            x_slope = np.linspace(valid[x].min(), valid[x].max(), 100)
            y_slope = 10 ** p(np.log10(x_slope))
            #ax.plot(x_slope, y_slope,color="#1f2933", lw=1.2, ls="--")
            corr = np.corrcoef(log_x, log_y)[0, 1]
            #ax.text(0.04, 0.95,f"r = {corr:.2f}",transform=ax.transAxes,va="top",fontsize=9)

        ax.set_title(f"{t} (n = {len(sub):,} runs)",loc="left",fontweight="bold")

        ax.grid(axis="y", color="#e4e8ed")
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)

        if logy:
            ax.set_yscale("log")
            ax.set_xscale("log")

        ax.set_xlabel(x)

    axes[0].set_ylabel(ylabel)


    handles = [
        Line2D(
            [], [], marker="o", linestyle="",
            color=age_colors[age],
            label=str(age),
            markersize=6
        )
        for age in age_bins
    ]

    fig.legend(
        handles=handles,
        title="Building age class",
        loc="upper right",
        ncol=min(len(age_bins), 6),
        frameon=False
    )

    fig.suptitle(
        title, x=0.01, ha="left",
        fontsize=14, fontweight="bold"
    )

    fig.tight_layout(rect=(0, 0, 1, 0.88))

# weather data

def load_TRY_files(try_path):
    dirs = {}
    for year in [2015, 2045]:
        for region in [f"{i:02d}" for i in range(16)]:
            year_str = f"TRY{year}"
            trgt_dir = try_path / year_str /str(region)
            files = trgt_dir.glob("*.dat")
            for file in list(files):
                weather_type = file.stem.split("_")[-1]
                weather = pd.read_csv(file, sep=r"\s+", skiprows=list(range(32)) + [33], skip_blank_lines=True,encoding="utf-8")
                weather.index = pd.to_datetime({"year":year, "month": weather["MM"],"day": weather["DD"],"hour": weather["HH"]-1})
                weather.index.name = "datetime"
                dirs.setdefault(year, {}).setdefault(region, {}).setdefault(weather_type, {})[file.stem] = weather

    weather_df = pd.concat(
        {
            (year, region, weather_type, file_id): df
            for year, regions in dirs.items()
            for region, types in regions.items()
            for weather_type, files in types.items()
            for file_id, df in files.items()
        },
        names=["year", "region", "weather_type", "file_id"]
    )

    mean_df = weather_df.groupby(level=["year", "region", "weather_type", "file_id"])["t"].mean().reset_index(name="mean_temp")
    return weather_df, mean_df

def plot_TRYs(weather_df, region):

    daily_temp =weather_df.groupby(level=["year", "region", "weather_type", "file_id"]).resample("D", level="datetime")["t"].mean().rename("t_mean").reset_index()
    daily_quantiles = daily_temp.groupby(["year", "region", "weather_type", "datetime"])["t_mean"].agg(
            q10=lambda x: x.quantile(0.10),
            median="median",
            mean="mean",
            q90=lambda x: x.quantile(0.90)).reset_index()

    region = region
    types = ["Jahr", "Somm", "Wint"]
    years = [2015, 2045]
    colors = {
        "Jahr": "#32a852",
        "Somm": "#de290d",
        "Wint": "#101dcc"
    }
    labels = {
        "Jahr": "Average",
        "Somm": "Hot",
        "Wint": "Cold"
    }
    fig, axes = plt.subplots(
        2, 1,
        figsize=(14, 8),
        sharex=True,
        sharey=True
    )
    linestyles = {
        "Jahr": "-",
        "Somm": "--",
        "Wint": ":"
    }
    for ax, year in zip(axes, years):
        for weather_type in types:
            sub = daily_quantiles.loc[
                (daily_quantiles["region"] == region) &
                (daily_quantiles["weather_type"] == weather_type) &
                (daily_quantiles["year"] == year)
            ].sort_values("datetime")
            x = sub["datetime"].dt.dayofyear.to_numpy()
            median = sub["median"].to_numpy(dtype=float)
            q10 = sub["q10"].to_numpy(dtype=float)
            q90 = sub["q90"].to_numpy(dtype=float)
            ax.fill_between(x,q10,q90,color=colors[weather_type],alpha=0.1,linewidth=0)
            ax.plot(
        x,
        median,
        color=colors[weather_type],
        linestyle=linestyles[weather_type],
        linewidth=1.5,
        label=labels[weather_type]
    )

        ax.set_title(
            str(year),
            loc="left",
            fontweight="bold"
        )

        ax.axhline(
            0,
            color="gray",
            linewidth=0.8,
            linestyle="--"
        )

        ax.grid(axis="y", color="#e4e8ed")
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_ylabel("Temperature °C")

    axes[-1].set_xlabel("Day of year")

    axes[0].legend(
        frameon=False,
        ncol=3,
        loc="upper right"
    )

    fig.suptitle(
        f"Daily mean temperatures CR {region}",
        x=0.125,
        ha="left",
        fontsize=15,
        fontweight="bold"
    )

    plt.tight_layout()
    plt.show()

def boxplot_TRY(mean_df):

    sns.set_theme(style="white")

    labels = {
        "Jahr": "Average",
        "Somm": "Hot",
        "Wint": "Cold"
    }

    g = sns.catplot(
        data=mean_df,
        x="region",
        y="mean_temp",
        hue="year",
        col="weather_type",
        col_order=["Jahr", "Somm", "Wint"],
        kind="box",
        height=5,
        aspect=1.2,
        palette=["#76B7AD", "#E99A7D"],
        sharey=True,
        fliersize=2.5
    )

    for ax, t in zip(g.axes.flat, ["Jahr", "Somm", "Wint"]):
        sub = mean_df[mean_df["weather_type"] == t]

        ax.set_title(
            f"{labels[t]} (n = {len(sub):,} files)",
            loc="left",
            fontweight="bold"
        )

        ax.grid(axis="y", color="#e4e8ed")
        ax.set_axisbelow(True)

        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(axis="x", length=0)

        plt.setp(
            ax.get_xticklabels(),
            rotation=45,
            ha="right"
        )

    g.set_axis_labels("Region", "Mean temperature °C")

    g.figure.suptitle(
        "Mean temperature by region, year and weather type",
        x=0.03,
        ha="left",
        fontweight="bold",
        fontsize=15
    )

    g.figure.subplots_adjust(top=0.85)
    plt.show()

# Appliances

def plot_appliance_ownership_total(manifest):
    manifest = manifest.copy()
    apls_per_run = manifest["appliances"].map(lambda s: ast.literal_eval(s) if isinstance(s, str) else s)
    assert manifest["n_apartments"].sum() == apls_per_run.apply(len).sum()
    apls_lists = apls_per_run.map(lambda full_list: [item for sub_list in full_list for item in sub_list])
    apl_df = pd.DataFrame({"buildingType": manifest["buildingType"], "appliance":apls_lists}).explode("appliance")
    owned = apl_df.groupby("appliance").size().sort_values()
    owned_rel = owned/apls_per_run.apply(len).sum()

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
    ax.set_xlabel("Share of Apartments")
    fig.tight_layout()

    return owned_rel

def plot_total_appliances_vs_metric(df,metric,ylabel=None,title=None,logy=False,x="mean_appliances_per_apartment"):
    fig, axes = plt.subplots(1, 4, figsize=(18, 5), sharey=True)
    for ax, t in zip(axes.flat, TYPES):
        sub = df.loc[df["buildingType"] == t]
        x_group = sub[x].round()
        params = sorted(x_group.dropna().unique())
        groups = [sub.loc[x_group == a, metric].dropna().values for a in params]

        ax.boxplot(
            groups,
            positions=range(len(params)),
            widths=0.6,
            patch_artist=True,
            whis=1.5,
            medianprops=dict(color="#1f2933", lw=1.3),
            boxprops=dict(
                facecolor=COLOR_DICT[t],
                alpha=0.55,
                edgecolor=COLOR_DICT[t]
            ),
            whiskerprops=dict(color=COLOR_DICT[t]),
            capprops=dict(color=COLOR_DICT[t]),
            flierprops=dict(
                marker=".",
                markersize=3,
                markeredgecolor="#6b7785",
                alpha=0.5
            )
        )

        ax.set_xticks(range(len(params)))
        ax.set_xticklabels([f"{p:g}" for p in params])
        ax.tick_params(axis="x", rotation=45)

        ax.set_title(f"{t} (n = {len(sub):,} runs)",loc="left",fontweight="bold")
        ax.set_xlabel("Appliances per apartment")
        ax.set_ylabel("")

        ax.grid(axis="y", color="#e4e8ed")
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)

        if logy:
            ax.set_yscale("log")

    axes[0].set_ylabel(ylabel if ylabel else metric)
    fig.suptitle(
        title if title else f"Total appliances vs {metric}",
        x=0.01,
        ha="left",
        fontsize=14,
        fontweight="bold"
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    plt.show()


def plot_group_share_vs_metric(df, group, metric="elec_mean_per_apartment",
                               ylabel="Mean electricity load per apartment in kW"):
    share_col = f"share_{group}"
    data = df.copy()
    bins = [-0.001, 0, 0.25, 0.5, 0.999999, 1.0]
    labels = ["0%", "0–25%", "25–50%", "50–100%", "100%"]
    data["share_bin"] = pd.cut(
        data[share_col],
        bins=bins,
        labels=labels,
        include_lowest=True
    )
    fig, axes = plt.subplots(
        1, 4,
        figsize=(18, 5),
        sharey=True
    )
    for ax, t in zip(axes.flat, TYPES):
        sub = data.loc[data["buildingType"] == t]

        available = [
            label for label in labels
            if (sub["share_bin"] == label).any()
        ]

        groups = [
            sub.loc[sub["share_bin"] == label, metric]
            .dropna().values
            for label in available
        ]

        ax.boxplot(
            groups,
            positions=range(len(available)),
            widths=0.6,
            patch_artist=True,
            whis=1.5,
            medianprops=dict(color="#1f2933", lw=1.3),
            boxprops=dict(
                facecolor=COLOR_DICT[t],
                alpha=0.55,
                edgecolor=COLOR_DICT[t]
            ),
            whiskerprops=dict(color=COLOR_DICT[t]),
            capprops=dict(color=COLOR_DICT[t]),
            flierprops=dict(
                marker=".",
                markersize=3,
                markeredgecolor="#6b7785",
                alpha=0.5,

            )
        )

        ax.set_xticks(range(len(available)))
        ax.set_xticklabels(available, rotation=35)
        ax.set_title(f"{t} (n = {len(sub):,} runs)", loc="left", fontweight="bold")
        ax.set_xlabel("Appliance group ownership fraction")
        ax.set_ylabel("")
        ax.grid(axis="y", color="#e4e8ed")
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel(ylabel)
    fig.suptitle(f"{group} appliance ownership vs. {metric}", x=0.01, ha="left", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    plt.show()

def plot_appliance_groupshare_2x5(df,metric="elec_max_per_apartment",ylabel="Maximum electricity load per apartment in kW"):
    data = df.copy()
    bins = [-0.001, 0, 0.25, 0.5, 0.999999, 1.0]
    labels = ["0%", "0–25%", "25–50%", "50–100%", "100%"]
    building_groups = ["SFH + TH", "MFH + AB"]
    appliance_groups = list(APPLIANCE_GROUPS.keys())
    group_colors = {"SFH + TH": "#8b865b","MFH + AB": "#926866"}
    fig, axes = plt.subplots(2, len(appliance_groups),figsize=(4 * len(appliance_groups), 9),sharey="row")

    for i, bgroup in enumerate(building_groups):
        for j, agroup in enumerate(appliance_groups):
            ax = axes[i, j]
            share_col = f"share_{agroup}"
            sub = data.loc[data["buildingGroup"] == bgroup, [share_col, metric]].copy()
            sub["share_bin"] = pd.cut(
                sub[share_col],
                bins=bins,
                labels=labels,
                include_lowest=True
            )

            available = [
    label for label in labels
    if sub.loc[sub["share_bin"] == label, metric].notna().any()
]
            positions = [labels.index(label) for label in available]
            groups = [sub.loc[sub["share_bin"] == label, metric].dropna().values for label in available]

            if len(groups) > 0:
                ax.boxplot(
                    groups,
                    positions=positions,
                    widths=0.6,
                    patch_artist=True,
                    whis=1.5,
                    medianprops=dict(color="#1f2933", lw=1.3),
                    boxprops=dict(
                        facecolor=group_colors[bgroup],
                        alpha=0.6,
                        edgecolor=group_colors[bgroup]
                    ),
                    whiskerprops=dict(color=group_colors[bgroup]),
                    capprops=dict(color=group_colors[bgroup]),
                    flierprops=dict(
                        marker=".",
                        markersize=2.5,
                        markeredgecolor="#6b7785",
                        alpha=0.45
                    )
                )

            counts = [
                sub.loc[sub["share_bin"] == label, metric].notna().sum()
                for label in labels
            ]

            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels([
                f"{label}\n(n={c})" for label, c in zip(labels, counts)
            ])

            ax.set_xlim(-0.5, len(labels) - 0.5)

            if j == 0:
                ax.set_ylabel(ylabel)
            else:
                ax.set_ylabel("")

            if i == 1:
                ax.set_xlabel("Ownership fraction")
            else:
                ax.set_xlabel("")

            if i == 0:
                ax.set_title(agroup, loc="left", fontweight="bold")

            if j == 0:
                n_runs = (data["buildingGroup"] == bgroup).sum()
                ax.set_ylabel(ylabel)

            else:
                ax.set_ylabel("")

            if i == 1:
                ax.set_xlabel("Ownership fraction")
            else:
                ax.set_xlabel("")

            ax.grid(axis="y", color="#e4e8ed")
            ax.set_axisbelow(True)
            ax.spines[["top", "right"]].set_visible(False)


    fig.suptitle(
        f"Appliance group ownership vs. {metric}",
        x=0.06,
        y=0.98,
        ha="left",
        fontsize=14,
        fontweight="bold"
    )

    fig.subplots_adjust(
        left=0.06,
        right=0.99,
        top=0.86,
        bottom=0.10,
        hspace=0.65,
        wspace=0.12
    )

    for j, agroup in enumerate(appliance_groups):
        axes[0, j].set_title(
            agroup,
            loc="left",
            fontweight="bold",
            fontsize=11,
            pad=10
        )

    for i, bgroup in enumerate(building_groups):
        n_runs = (data["buildingGroup"] == bgroup).sum()

        row_pos = axes[i, 0].get_position()

        fig.text(
            row_pos.x0,
            row_pos.y1 + (0.045 if i == 0 else 0.025),
            f"{bgroup} (n = {n_runs:,} runs)",
            ha="left",
            va="bottom",
            fontsize=12,
            fontweight="bold"
        )

    plt.show()

def plot_simultaneity_effect(df):
    plt.figure(figsize=(8, 5))
    sns.boxplot(
        data=df,
        x="n_apartments",
        y="peak_to_average",
        color="#74a9a3",
        showfliers=False
    )
    plt.xlabel("Number of apartments")
    plt.ylabel("Peak-to-average load ratio")
    plt.title("Effect of apartment count on electricity load diversity")
    plt.grid(axis="y", color="#e4e8ed")
    plt.tight_layout()
    plt.show()

def plot_appliance_heatmap(df,share_df):
    metrics = ["elec_mean","elec_max","elec_min","elec_mean_01_05","elec_p95","elec_p05"]

    corr = df[list(share_df.columns) + metrics].corr(method="spearman")
    plt.figure(figsize=(8, 5))
    sns.heatmap(corr.loc[list(share_df.columns), metrics],annot=True,fmt=".2f",cmap="RdBu_r",center=0,vmin=-1,vmax=1)
    plt.title("Appliance group ownership vs. electricity demand")
    plt.tight_layout()
    plt.show()

APPLIANCE_GROUPS = {
    "Continuous": [
        "CHEST_FREEZER", "FRIDGE", "FRIDGE_FREEZER", "UPRIGHT_FREEZER",
    ],
    "Entertainment": [
        "TV1", "TV2","TV3","VCR_DVD","RECEIVER", "CD_PLAYER","HIFI"
    ],
    "Cleaning": [
        "DISH_WASHER", "TUMBLE_DRYER", "WASHER_DRYER", "WASHING_MACHINE", "IRON", "VACUUM"
    ],
    "Cooking": [
        "HOB", "KETTLE", "MICROWAVE", "OVEN", "SMALL_COOKING"
    ],
    "Other": [
        "CLOCK", "PHONE","PRINTER","PC", "FAX", "ANSWER_MACHINE"
    ]
}

def calculate_group_shares(apartments):
    appliance_to_group = {appliance: group for group, appliances in APPLIANCE_GROUPS.items() for appliance in appliances}
    if isinstance(apartments, str):
        apartments = ast.literal_eval(apartments)
    n_apartments = len(apartments)
    counts = Counter()
    for apartment in apartments:
        for appliance in set(apartment):
            group = appliance_to_group.get(appliance)
            if group is not None:
                counts[group] += 1
    shares = {}
    for group, appliances in APPLIANCE_GROUPS.items():
        maximum = n_apartments * len(appliances)
        shares[f"share_{group}"] = (counts[group] / maximum if maximum > 0 else float("nan"))

    return pd.Series(shares)

# Correlation
def avg_corr(data, offday=2):
    data_offday = data.xs(offday, level="offday")
    #fisher z transform
    z = np.arctanh(data_offday.clip(-1 + 1e-12, 1 - 1e-12))
    mean_z = z.groupby(level=["buildingType","freq", "resid", "variable"],sort=False).mean()
    mean_corr_fisher = np.tanh(mean_z)
    std_corr = data_offday.groupby(level=["buildingType","freq", "resid", "variable"], sort=False).std()
    n_corr = data_offday.groupby(level=["buildingType","freq", "resid", "variable"], sort=False).count()
    mean_corr = mean_corr_fisher.reindex(data.columns, level="variable")
    std_corr = std_corr.reindex(data.columns, level="variable")
    n_corr = n_corr.reindex(data.columns, level="variable")
    return mean_corr, std_corr, n_corr

def avg_partial_corr(data, offday=2):
    data = data.xs(offday, level="offday")
    #fisher z transform
    z = np.arctanh(data["r"].clip(-1 + 1e-12, 1 - 1e-12))
    levels = ["buildingType", "freq", "resid", "comparison"]
    grouped = z.groupby(level=levels, sort=False)
    result = pd.DataFrame({
        "r_mean": np.tanh(grouped.mean()),
        "r_std_z": grouped.std(),
        "n_runs": grouped.count(),
    })
    return result


def plot_corr_matrices(mean_corr, std_corr, n_corr, freq="1min", resid=False, show_std=False):
    levels = ("freq", "resid")
    selected_mean = mean_corr.xs((freq, resid), level=levels)
    selected_std = std_corr.xs((freq, resid), level=levels)
    selected_n = n_corr.xs((freq, resid), level=levels)

    building_types = selected_mean.index.get_level_values("buildingType").unique()

    labels = {
        "Electricity Load": "Electricity",
        "Hot Water Load": "Hot Water",
        "Heating Load": "Heating",
        "Occupancy Home Active": "Home Active",
        "Occupancy Home Not Active": "Home Inactive",
        "T": "Temperature",
    }

    n = len(building_types)
    ncols = min(2, n)
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 5.5 * nrows), squeeze=False, layout="constrained")

    axes = axes.flatten()
    for ax, building_type in zip(axes, building_types):
        mean = selected_mean.xs(building_type, level="buildingType")
        std = selected_std.xs(building_type, level="buildingType")
        counts = selected_n.xs(building_type, level="buildingType")

        variables = mean.columns
        mean = mean.reindex(index=variables, columns=variables)
        std = std.reindex(index=variables, columns=variables)
        counts = counts.reindex(index=variables, columns=variables)

        n_min = int(counts.min().min())
        n_max = int(counts.max().max())

        n_label = (f"n = {n_min}" if n_min == n_max else f"n = {n_min}–{n_max}")

        annot = pd.DataFrame("", index=variables, columns=variables)

        for row in variables:
            for col in variables:
                r = mean.loc[row, col]
                s = std.loc[row, col]

                if pd.isna(r):
                    continue

                if show_std and pd.notna(s):
                    annot.loc[row, col] = f"{r:.2f}\n±{s:.2f}"
                else:
                    annot.loc[row, col] = f"{r:.2f}"

        mask = np.triu(np.ones(mean.shape, dtype=bool), k=1)

        mean = mean.rename(index=labels, columns=labels)
        sns.heatmap(
            mean,
            mask=mask,
            annot=annot.to_numpy(),
            fmt="",
            ax=ax,
            cmap="RdBu_r",
            vmin=-1,
            vmax=1,
            center=0,
            square=True,
            cbar=False,
            linewidths=0.5,
            linecolor="white",
            annot_kws={"fontsize": 9},
        )

        ax.set_title(f"{building_type} ({n_label})", fontsize=13, fontweight="bold", pad=12)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(axis="both", length=0, labelsize=10)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=35, ha="right")

        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

    for ax in axes[n:]:
        ax.set_visible(False)

    sm = plt.cm.ScalarMappable(cmap="RdBu_r", norm=plt.Normalize(-1, 1))
    fig.colorbar(sm, ax=axes[:n].tolist(), shrink=0.75, label="Mean Spearman correlation")
    fig.suptitle(f"Average Correlation Matrices | {freq} | Residuals: {resid}", fontsize=15, fontweight="bold", )

from itertools import product
def lagged_corr(data,pairs,freq="1min",max_lag="60min",resid=False,Offday=2,method="spearman"):
    df = prepare_corr_data(data, freq, resid, Offday)
    step = pd.to_timedelta(freq)
    max_lag_steps = int(pd.to_timedelta(max_lag) / step)

    results = []
    for (x, y), lag in product(pairs,range(-max_lag_steps, max_lag_steps + 1)):
        shifted_y = df[y].shift(-lag)
        valid = df[x].notna() & shifted_y.notna()
        n = valid.sum()

        r = (df.loc[valid, x].corr(shifted_y.loc[valid],method=method) if n >= 3 else np.nan)

        results.append({
            "x": x,
            "y": y,
            "lag_min": lag * step.total_seconds() / 60,
            "r": r,
            "n": n,
        })

    return pd.DataFrame(results)

import matplotlib.pyplot as plt
import seaborn as sns

def plot_lagged_corr(data):
    fig, ax = plt.subplots(figsize=(10, 5))

    data = data.copy()
    data["pair"] = data["x"] + " -> " + data["y"]

    sns.lineplot(
        data=data,
        x="lag_min",
        y="r",
        hue="pair",
        palette="Dark2",
        ax=ax,
    )

    ax.axvline(0, color="black", linestyle="--", alpha=0.5)
    ax.axhline(0, color="grey", linewidth=0.8)

    ax.set_xlabel("Lag (minutes)")
    ax.set_ylabel("Spearman correlation")
    ax.set_ylim(-1, 1)
    ax.set_title("Lagged Cross-Correlations")

    ax.legend(title="Variable pair", bbox_to_anchor=(1.02, 1))
    ax.grid(alpha=0.2)

    return fig, ax