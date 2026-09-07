from pathlib import Path
import json
import pandas as pd
import ast
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
RESULTS_DIR = r"DataPreparation\Generation\Generation"
test_path = r"Generation/Generation/run_000013.h5"
MANIFEST_PATH = r"DataPreparation/Generation/manifest.parquet"


def manifest_distribution(manifest: pd.DataFrame):
    manifest = manifest.copy()
    manifest["n_persons"] = manifest["n_persons"].apply( lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    manifest["n_persons_sum"] = [sum(x) for x in manifest["n_persons"]]
    manifest["area_per_person"] = manifest["a_ref"]/manifest["n_persons_sum"]
    #General Checks
    print("Overview")
    print("="*80)
    print(f"Rows: {len(manifest):,}")
    print(f"Unique buildings: {manifest['a_ref'].nunique():,}")
    print(f"Unique runs: {manifest['seed'].nunique():,}")
    print("MISSING VALUES")
    print("=" * 80)
    print(manifest.isna().sum().sort_values(ascending=False))

    #Check Categorial Distribution over all Runs
    categorical_cols = [
    "buildingType",
    "surrounding",
    "buildingAgeBin",
    "year_type",
    "future",
    "climateRegion",
    ]

    for col in categorical_cols:
        if col not in manifest.columns:
            continue

        print("=" * 80)
        print(f"DISTRIBUTION: {col}")
        print("=" * 80)

        counts = manifest[col].value_counts(dropna=False).sort_index()
        percentages = manifest[col].value_counts(
            normalize=True,
            dropna=False
        ).sort_index() * 100

        distribution = pd.DataFrame({
            "count": counts,
            "percent": percentages
        })

        print(distribution)
        print()

    #Check Categorial Distribution for unique buildings only

    building_df = (manifest.drop_duplicates("a_ref").copy())
    print("=" * 80)
    print("BUILDING-LEVEL DATA")
    print("=" * 80)
    print(f"Simulation rows: {len(manifest):,}")
    print(f"Distinct building scenarios: {len(building_df):,}")
    print()

    for col in categorical_cols:
        if col not in building_df.columns:
            continue

        print("=" * 80)
        print(f"BUILDING-LEVEL DISTRIBUTION: {col}")
        print("=" * 80)

        counts = building_df[col].value_counts(dropna=False).sort_index()
        percentages = building_df[col].value_counts(
            normalize=True,
            dropna=False
        ).sort_index() * 100

        print(pd.DataFrame({"count": counts,"percent": percentages}))
        print()

    #Plots
    #histogramm for distribution variabbles
    distribution_cols = [
    "a_ref",
    "n_apartments",
    "n_persons_sum",
    ]

    for col in distribution_cols:
        if col not in building_df.columns:
            continue

        fig1, ax1 = plt.subplots(2,2, figsize=(12,8))

        for ax, bt in zip(ax1.flat, manifest["buildingType"].unique()):
            x = manifest.loc[manifest["buildingType"]== bt, col]

            ax.hist(x, bins=25)
            ax.set_title(bt)
            ax.set_xlabel(col)
            ax.set_ylabel("Buildings")
        plt.tight_layout()
        plt.show()

    fig1, ax1 = plt.subplots()
    x = manifest.loc[:, "n_persons_sum"]

    ax1.hist(x, bins=20)
    ax1.set_title("Total occ distributio")
    ax1.set_xlabel("number of occs")
    ax1.set_ylabel("Buildings")
    plt.tight_layout()
    plt.show()

    fig1, ax1 = plt.subplots()
    x = manifest.loc[:, "a_ref"]
    ax1.hist(x, bins=20)
    ax1.set_title("Total area distribution")
    ax1.set_xlabel("area in m²")
    ax1.set_ylabel("Buildings")
    plt.tight_layout()
    plt.show()

    fig1, ax1 = plt.subplots()
    x = manifest.loc[:, "n_apartments"]
    ax1.hist(x, bins=20)
    ax1.set_title("distribution of apartments")
    ax1.set_xlabel("n apartments")
    ax1.set_ylabel("Buildings")
    plt.tight_layout()
    plt.show()

    fig1, ax1 = plt.subplots()
    x = manifest.loc[:, "area_per_person"]
    ax1.hist(x, bins=20)
    ax1.set_title("distribution of area per person")
    ax1.set_xlabel("area in m²/person")
    ax1.set_ylabel("Buildings")
    plt.tight_layout()
    plt.show()


def comp_manifests(planned_manifest: pd.DataFrame, realised_manifest: pd.DataFrame)->None:
    planned_manifest = planned_manifest.copy()
    realised_manifest = realised_manifest.copy()

    print(f"Planned: {len(planned_manifest)}")
    print(f"Realized: {len(realised_manifest)}")

    #check for missing rows

    real_missing = planned_manifest["seed"].isin(realised_manifest["seed"])
    missing_cols = planned_manifest[~real_missing]
    print(f"Missing Cols: {len(missing_cols)}")
    print(f"{missing_cols.index}")



    planned_cols_with_missing = planned_manifest[real_missing]
    #check if manifest values got changed in the generation

    assert (planned_cols_with_missing["seed"] == realised_manifest["state_seed"]).all()
    assert (planned_cols_with_missing["climateRegion"] == realised_manifest["resolved_climateRegion"]).all()
    realised_manifest_bench = realised_manifest.drop(columns=["weatherFilepath", "weatherID", "resolved_climateRegion","resolved_latitude","resolved_longitude","state_seed","apartment_seeds","maxLoadViolation_kW","runtime_seconds"])
    assert len(realised_manifest_bench) == len(planned_cols_with_missing)
    realised_manifest_bench["run_id"] = planned_cols_with_missing["run_id"].to_numpy()
    realised_manifest_bench["building_id"] = planned_cols_with_missing["building_id"].to_numpy()
    planned_manifest_bench = planned_cols_with_missing#.drop(columns=["building_id", "run_id"])
    string_cols = (planned_manifest_bench.select_dtypes(include="string").columns.union(realised_manifest_bench.select_dtypes(include="string").columns))

    for col in string_cols:
        planned_manifest_bench[col] = planned_manifest_bench[col].astype("string")
        realised_manifest_bench[col] = realised_manifest_bench[col].astype("string")

    realised_manifest_bench = realised_manifest_bench[planned_manifest_bench.columns].reset_index(drop=True)
    planned_manifest_bench = planned_manifest_bench.reset_index(drop=True)

    print(f"Planned and Realized Values are identical: {realised_manifest_bench.equals(planned_manifest_bench)}")
    print(f"Planned and Realized Differences: {realised_manifest_bench.compare(planned_manifest_bench)}")

    # distribution check planned
    print("overview planned" + "=" *80)
    manifest_distribution(planned_cols_with_missing)
    # distribution check realized
    print("overview realized" + "=" *80)
    manifest_distribution(realised_manifest)

    #check unique generation distributions for realized
    for reg in sorted(realised_manifest["climateRegion"].unique()):
        print(realised_manifest[realised_manifest["climateRegion"] == reg].groupby(["year_type","future"])["weatherID"].value_counts(normalize=True, dropna=False))


def check_timeseries(resultsdir) -> pd.DataFrame:

    resultsdir = Path(resultsdir)
    completed_runs = sorted(resultsdir.glob("run_*.h5"))

    load_domains = {
        "elec": "Electricity Load",
        "dhw": "Hot Water Load",
        "sh": "Heating Load",
    }

    dt_hours = 1 / 60  # minutely resolution

    rows = []
    for run in tqdm(
        completed_runs,
        desc="Checking timeseries",
        unit="run",
    ):

        ts = pd.read_hdf(run, key="timeseries")
        params = pd.read_hdf(run, key="params").iloc[0]
        n_persons = params["n_persons"]

        if isinstance(n_persons, str):
            n_persons = ast.literal_eval(n_persons)

        total_persons = sum(n_persons)
        a_ref = float(params["a_ref"])

        numeric_ts = ts.select_dtypes(include="number")

        res = {
            "run_id": run.stem,
            "nan_vals": ts.isna().any().any(),
            "neg_load_vals": (ts[["Electricity Load","Hot Water Load","Heating Load",]] < 0).any().any(),
            "total_persons": total_persons,
            "a_ref": a_ref,
        }


        for prefix, col in load_domains.items():
            load = ts[col]
            annual_energy = load.sum() * dt_hours
            res.update({
                f"{prefix}_sum_yearly": annual_energy,
                f"{prefix}_sum_yearly_per_occ": annual_energy / total_persons,
                f"{prefix}_sum_yearly_per_a_ref": annual_energy / a_ref,
                f"{prefix}_max":load.max(),
                f"{prefix}_min":load.min(),
                f"{prefix}_mean":load.mean(),
                f"{prefix}_p95":load.quantile(0.95),
                f"{prefix}_p99": load.quantile(0.99),
            })

        if "T" in ts.columns:
            temp = ts["T"]

            res.update({
                "temperature_mean": temp.mean(),
                "temperature_min":temp.min(),
                "temperature_max":temp.max(),
                "temperature_p05":temp.quantile(0.05),
                "temperature_p95":temp.quantile(0.95),
            })

        for col in [
            "buildingType",
            "buildingAgeBin",
            "surrounding",
            "climateRegion",
            "year_type",
            "future",
            "seed",
            "state_seed",
        ]:
            if col in params.index:
                res[col] = params[col]

        rows.append(res)

    return pd.DataFrame(rows)

def check_duplicate_timeseries(manifest):
    duped_list = manifest["apartment_seeds"].duplicated(keep=False).any()
    exploded = manifest['apartment_seeds'].explode()
    duped_single = exploded[exploded.duplicated(keep=False)].unique()
    print(f"Duplicated Seed Lists: {duped_list}")
    print(f"Duplicated Seeds: {duped_single}")