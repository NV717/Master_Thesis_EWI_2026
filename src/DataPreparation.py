from pathlib import Path

import pandas as pd
import ast
import numpy as np
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import pingouin as pg

RESULTS_DIR = r"DataPreparation/Generation/Generation"
test_path = r"Generation/Generation/run_000013.h5"
MANIFEST_PATH = r"DataPreparation/Generation/manifest.parquet"

def get_realized_manifest(results_dir: Path):
    completed_runs = results_dir.glob("run_*.h5")
    dfs = [pd.read_hdf(run, key="params") for run in tqdm(completed_runs)]
    realised_manifest = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
    realised_manifest = realised_manifest.sort_values(by=["seed"]).reset_index(drop=True)
    return realised_manifest

def add_appliance_shares(manifest):
    manifest = manifest.copy()
    apls_per_run = manifest["appliances"].map(lambda s: ast.literal_eval(s) if isinstance(s, str) else s)
    assert manifest["n_apartments"].sum() == apls_per_run.apply(len).sum()
    apls_lists = apls_per_run.map(lambda full_list: [item for sub_list in full_list for item in sub_list])
    unique_apls = sorted(list(apls_lists.explode().unique()))
    shares = pd.DataFrame(
        {f"apl_{a}": apls_lists.apply(lambda fulllist: sum(a == apl for apl in fulllist)) / apls_per_run.apply(len) for a in unique_apls}, index=manifest.index,
    )

    return  pd.concat([manifest, shares], axis=1), unique_apls

def add_appliance_counts(df):
    df = df.copy()
    apls_per_run = df["appliances"].map(lambda s: ast.literal_eval(s) if isinstance(s, str) else s)
    df["appliances_per_apartment"] = apls_per_run.map(lambda apartments: [len(appliances) for appliances in apartments])
    df["total_appliances"] = df["appliances_per_apartment"].map(sum)
    df["mean_appliances_per_apartment"] = df["total_appliances"] / df["n_apartments"]
    return df

def per_apartment_loads(df):
    df = df.copy()
    df["elec_min_per_apartment"] =df["elec_min"] / df["n_apartments"]
    df["elec_max_per_apartment"] =df["elec_max"] / df["n_apartments"]
    df["elec_mean_per_apartment"] =df["elec_mean"] / df["n_apartments"]
    df["elec_mean_01_05_per_apartment"] =df["elec_mean_01_05"] / df["n_apartments"]
    df["elec_p05_per_apartment"] =df["elec_p05"] / df["n_apartments"]
    df["elec_p10_per_apartment"] =df["elec_p10"] / df["n_apartments"]
    df["elec_p95_per_apartment"] =df["elec_p95"] / df["n_apartments"]
    df["elec_p99_per_apartment"] =df["elec_p99"] / df["n_apartments"]
    df["elec_sum_yearly_per_apartment"] =df["elec_sum_yearly"] / df["n_apartments"]
    df["peak_to_average"] = df["elec_max"] / df["elec_mean"]
    df["buildingGroup"] = df["buildingType"].replace({
        "SFH": "SFH + TH",
        "TH": "SFH + TH",
        "MFH": "MFH + AB",
        "AB": "MFH + AB"
    })
    return df


def clean_list_cols(realized_manifest: pd.DataFrame, planned_manifest: pd.DataFrame):
    # sum persons
    realized_manifest["n_persons"] = realized_manifest["n_persons"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    realized_manifest["n_persons_sum"] = realized_manifest["n_persons"].apply(sum)
    #get unique appliances
    realized_manifest["appliances"] = realized_manifest["appliances"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    realized_manifest["appliances_unique"] = realized_manifest["appliances"].apply(lambda x: list(dict.fromkeys(item for sublist in x for item in sublist)))
    # get building id back
    realized_manifest = realized_manifest.merge(planned_manifest[["seed", "building_id", "run_id"]], on="seed",how="left")
    return realized_manifest


def manifest_distribution(manifest: pd.DataFrame):
    manifest = manifest.copy()
    manifest["n_persons"] = manifest["n_persons"].apply( lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    manifest["n_persons_sum"] = [sum(x) for x in manifest["n_persons"]]
    manifest["area_per_person"] = manifest["a_ref"]/manifest["n_persons_sum"]
    #General Checks
    print("Overview")
    print("="*80)
    print(f"Rows: {len(manifest):,}")
    print(f"Unique areas (a_ref): {manifest['a_ref'].nunique():,}")
    print(f"Unique buildings: {manifest['building_id'].nunique():,}")
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

    building_df = (manifest.drop_duplicates("building_id").copy())
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

    planned_cols_with_missing = planned_manifest[
        planned_manifest["seed"].isin(realised_manifest["state_seed"])
    ].copy()

    planned_cols_with_missing = (
        planned_cols_with_missing
        .sort_values("seed")
        .reset_index(drop=True)
    )

    realised_manifest = (
        realised_manifest
        .sort_values("state_seed")
        .reset_index(drop=True)
    )

    assert len(planned_cols_with_missing) == len(realised_manifest)

    assert (
            planned_cols_with_missing["seed"].to_numpy()
            == realised_manifest["state_seed"].to_numpy()
    ).all()

    assert (
            planned_cols_with_missing["climateRegion"].to_numpy()
            == realised_manifest["resolved_climateRegion"].to_numpy()
    ).all()


    realised_manifest_bench = realised_manifest.drop(columns=["weatherFilepath", "weatherID", "resolved_climateRegion","resolved_latitude","resolved_longitude","state_seed","apartment_seeds","maxLoadViolation_kW","runtime_seconds", "appliances", "appliances_unique"])
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


OCC_COL = "Occupancy Home Active"
OFFDAY_COL = "Off Day"
WINTER_MONTHS = [10, 11, 12, 1, 2, 3]
PARAM_COLS = ["buildingType", "buildingAgeBin", "surrounding", "climateRegion","year_type", "future", "seed", "state_seed"]
DT_HOURS = 1 / 60

def _run_info(run, ts, params, load_cols, total_persons, a_ref) -> dict:
    occ = ts[OCC_COL]
    g = occ.groupby(occ.index.date)
    info = {
        "run_id": run.stem,
        "nan_vals": ts.isna().any().any(),
        "neg_load_vals": (ts[load_cols] < 0).any().any(),
        "total_persons": total_persons,
        "a_ref": a_ref,
        "occ_days_never_zero": int((g.min() > 0).sum()),
        "occ_days_always_zero": int((g.max() == 0).sum()),
    }
    if "T" in ts.columns:
        temp = ts["T"]
        info.update({
            "temperature_mean": temp.mean(),
            "temperature_min": temp.min(),
            "temperature_max": temp.max(),
            "temperature_p05": temp.quantile(0.05),
            "temperature_p95": temp.quantile(0.95),
        })
    info.update({c: params[c] for c in PARAM_COLS if c in params.index})
    return info

def _yearly_stats(load, occ, total_persons, a_ref) -> dict:
    night = load[(load.index.hour >= 1) & (load.index.hour < 5)]
    annual_energy = load.sum() * DT_HOURS
    n_days = load.index.normalize().nunique()
    res = {
        "sum_yearly": annual_energy,
        "sum_yearly_per_occ": annual_energy / total_persons,
        "sum_yearly_per_a_ref": annual_energy / a_ref,
        "max": load.max(),
        "min": load.min(),
        "mean": load.mean(),
        "p05": load.quantile(0.05),
        "p10": load.quantile(0.10),
        "p95": load.quantile(0.95),
        "p99": load.quantile(0.99),
        "mean_01_05": night.mean(),
        "load_factor": load.mean() / load.max() if load.max() > 0 else np.nan,
    }

    half = load.resample("30min").mean()
    winter = half.index.month.isin(WINTER_MONTHS)
    for label, series in {"": half, "_winter": half[winter], "_summer": half[~winter]}.items():
        res[f"tou{label}"], res[f"tou{label}_share"] = tou_parameter(series)

    unocc = peaks_without_occupancy(load, occ, prominence=0.075, plat_min_abs=0.05)
    res.update(unocc)

    return res

def _daily_stats(load, offday) -> list[dict]:
    g = load.groupby(load.index.date)
    d_max, d_mean = g.max(), g.mean()
    pps = peak_plateau_ramp_stats(load)
    g_without_zero = load.where(load != 0).groupby(load.index.date)
    all_daily = {
        "max": d_max,
        "min": g.min(),
        "mean": d_mean,
        "max_nz": g_without_zero.max(),
        "min_nz": g_without_zero.min(),
        "mean_nz": g_without_zero.mean(),
        "p05_nz": g.quantile(0.05),
        "p10_nz": g.quantile(0.10),
        "p95_nz": g.quantile(0.95),
        "p99_nz": g.quantile(0.99),
        "p05": g.quantile(0.05),
        "p10": g.quantile(0.10),
        "p95": g.quantile(0.95),
        "p99": g.quantile(0.99),
        "load_factor": (d_mean / d_max).where(d_max > 0),
        **{k: v for k, v in pps.items()},
    }

    rows = []
    for day_type, mask in {"all": None, "weekday": ~offday, "offday": offday}.items():
        row = {"day_type": day_type}
        for name, s in all_daily.items():
            if mask is not None:
                s = s[mask.loc[s.index]]
            row[f"{name}_avg"] = s.mean()
            row[f"{name}_std"] = s.std()
        row["active_days"] = len(pps) if mask is None else int(mask.loc[pps.index].sum())
        rows.append(row)
    return rows

def _hourly_profile(load, offday_minutes) -> list[dict]:
    groups = {
        "all": load,
        "weekday": load[~offday_minutes],
        "offday": load[offday_minutes],
    }
    rows = []
    for day_type, s in groups.items():
        profile = s.groupby(s.index.hour).mean()
        rows.append({"day_type": day_type, **{f"h{h:02d}": v for h, v in profile.items()}})
    return rows

from itertools import product

def _correlation_stats(ts,freqs=("1min", "15min", "1h"),offdays=(2,),residuals=(False, True),method="spearman"):
    correlations = []
    partial_correlations = []

    for freq, offday, resid in product(freqs, offdays, residuals):

        corr, partial = corr_table(ts,freq=freq,method=method,resid=resid,Offday=offday,partial=True)
        key = {"freq": freq,"offday": offday,"resid": resid}

        corr = corr.copy()
        corr.index.name = "variable"

        corr = pd.concat({tuple(key.values()): corr},names=list(key),)

        correlations.append(corr)

        partial = pd.concat({tuple(key.values()): partial},names=list(key),)

        partial_correlations.append(partial)

    return pd.concat(correlations),pd.concat(partial_correlations)

def _autocorrelation_stats(
    ts,
    freqs=("1min","15min", "1h"),
    offdays=(2,),
    residuals=(False, True),
    max_lag="48h",
    method="spearman",
):
    autocorrelations = []

    for freq, offday, resid in product(
        freqs, offdays, residuals
    ):
        acf = autocorr_table(
            ts,
            freq=freq,
            max_lag=max_lag,
            resid=resid,
            Offday=offday,
        )

        key = {
            "freq": freq,
            "offday": offday,
            "resid": resid,
        }

        acf = pd.concat(
            {tuple(key.values()): acf},
            names=list(key),
        )

        autocorrelations.append(acf)

    return pd.concat(autocorrelations)


import pickle
from joblib import Parallel, delayed

LOAD_DOMAINS = {
    "elec": "Electricity Load",
    "dhw": "Hot Water Load",
    "sh": "Heating Load",
}


def process_run(run: Path, correlations: bool = True) -> dict:
    ts = pd.read_hdf(run, key="timeseries")
    params = pd.read_hdf(run, key="params").iloc[0]

    n_persons = params["n_persons"]
    if isinstance(n_persons, str):
        n_persons = ast.literal_eval(n_persons)
    total_persons = sum(n_persons)
    a_ref = float(params["a_ref"])

    offday_minutes = ts[OFFDAY_COL].astype(bool)
    offday = offday_minutes.groupby(ts.index.date).max()
    occ = ts[OCC_COL]

    out = {
        "runs": _run_info(run, ts, params, list(LOAD_DOMAINS.values()), total_persons, a_ref),
        "yearly": [],
        "daily": [],
        "hourly": [],
    }
    for domain, col in LOAD_DOMAINS.items():
        load = ts[col]
        key = {"run_id": run.stem, "domain": domain}
        out["yearly"].append({**key, **_yearly_stats(load, occ, total_persons, a_ref)})
        out["daily"].extend({**key, **row} for row in _daily_stats(load, offday))
        out["hourly"].extend({**key, **row} for row in _hourly_profile(load, offday_minutes))

    if correlations:
        corr, partial = _correlation_stats(ts)

        out["correlations"] = corr
        out["partial_correlations"] = partial

        out["autocorrelations"] = _autocorrelation_stats(ts)
    return out


def process_run_cached(run: Path, cache_dir, correlations: bool = True) -> dict:
    cache_file = Path(cache_dir) / f"{run.stem}.pkl" if cache_dir else None
    if cache_file is not None and cache_file.exists():
        with open(cache_file, "rb") as f:
            cached = pickle.load(f)
        if not correlations or "correlations" in cached:
            return cached
    try:
        out = process_run(run, correlations)
    except Exception as e:
        return {"error": {"run_id": run.stem, "error": repr(e)}}
    if cache_file is not None:
        with open(cache_file, "wb") as f:
            pickle.dump(out, f)
    return out


def check_timeseries(resultsdir, n_jobs=-1, cache_dir=None, correlations=False) -> dict[str, pd.DataFrame]:
    resultsdir = Path(resultsdir)
    completed_runs = sorted(resultsdir.glob("run_*.h5"))
    if cache_dir is not None:
        Path(cache_dir).mkdir(parents=True, exist_ok=True)

    results = Parallel(n_jobs=n_jobs, return_as="generator")(
        delayed(process_run_cached)(run, cache_dir, correlations) for run in completed_runs
    )
    results = list(tqdm(results, total=len(completed_runs), unit="run"))

    ok = [r for r in results if "error" not in r]
    failed = [r["error"] for r in results if "error" in r]

    output= {
        "runs": pd.DataFrame([r["runs"] for r in ok]),
        "yearly": pd.DataFrame([row for r in ok for row in r["yearly"]]),
        "daily": pd.DataFrame([row for r in ok for row in r["daily"]]),
        "hourly": pd.DataFrame([row for r in ok for row in r["hourly"]]),
        "failed": pd.DataFrame(failed),
    }

    if ok and correlations:
        output["correlations"] = pd.concat({r["runs"]["run_id"]: r["correlations"] for r in ok},names=["run_id"])
        output["partial_correlations"] = pd.concat({r["runs"]["run_id"]: r["partial_correlations"]for r in ok},names=["run_id"])
        output["autocorrelations"] = pd.concat({r["runs"]["run_id"]: r["autocorrelations"]for r in ok},names=["run_id"])

    else:
        output["correlations"] = pd.DataFrame()
        output["partial_correlations"] = pd.DataFrame()
        output["autocorrelations"] = pd.DataFrame()

    return output

def tou_parameter(half_hourly):
    g = half_hourly.groupby(half_hourly.index.date)
    active = g.max() > 0
    peak_ts = g.idxmax()[active]
    if peak_ts.empty:
        return np.nan, np.nan
    j = peak_ts.dt.hour * 2 + peak_ts.dt.minute // 30 + 1
    mode = j.mode().min()
    return mode, (j == mode).mean()

def find_peak_plats(data, prominence=0.075, plat_tolerance=0.01, plat_min_level=0.10,plat_min_abs=0.05, plat_min_steps=10):
    peak = data.max()
    peak_idx, _ = find_peaks(data.values, prominence=prominence * peak, height=plat_min_abs)
    rol_max = data.rolling(plat_min_steps, center=True,min_periods=1).max()
    rol_min = data.rolling(plat_min_steps, center=True,min_periods=1).min()
    rol_mean = data.rolling(plat_min_steps, center=True,min_periods=1).mean()
    min_level = max(peak * plat_min_level,plat_min_abs)
    plat = (((rol_max - rol_min) <= rol_mean * plat_tolerance) & (rol_min > min_level))
    block = (plat != plat.shift()).cumsum()
    plat_len = plat.groupby(block).transform("size")
    plat_valid = plat & (plat_len >= plat_min_steps)
    return peak_idx, plat_valid, block, min_level

def peak_plateau_ramp_stats(load,freq = "1min",prominence = 0.075,  plat_tolerance = 0.01,      plat_min_level = 0.10,plat_min_abs = 0.05,    plat_min_steps = 10,min_day_level=0.05):

    data = load.resample(freq).mean()
    peak = data.max()
    dates = data.index.date

    day_max = data.groupby(dates).max()
    active_days = day_max.index[day_max >= max(min_day_level * peak, plat_min_abs)]

    peak_idx, plat_valid, block, _ = find_peak_plats(data, prominence, plat_tolerance, plat_min_level, plat_min_abs, plat_min_steps)
    peaks = (pd.Series(dates[peak_idx]).value_counts().reindex(active_days, fill_value=0).rename("peaks"))
    runs = (
        pd.DataFrame({"valid": plat_valid, "t": data.index})
        .groupby(block.values)
        .agg(valid=("valid", "first"), start=("t", "first"), n=("valid", "size"))
    )
    runs = runs[runs["valid"]]
    plateaus = runs.groupby(runs["start"].dt.date)["n"].agg(plateaus="size", plateau_dur_mean="mean", plateau_dur_max="max")

    ramp = data.diff()
    ramps = pd.DataFrame({
        "ramp_abs_mean": ramp.abs().groupby(dates).mean(),
        "ramp_abs_p95": ramp.abs().groupby(dates).quantile(0.95),
        "ramp_up_max": ramp.groupby(dates).max(),
        "ramp_down_max": ramp.groupby(dates).min(),
    })

    daily = pd.concat([peaks, plateaus.reindex(active_days), ramps.reindex(active_days)], axis=1)
    daily["plateaus"] = daily["plateaus"].fillna(0)
    return daily

def peaks_without_occupancy(load, occ, freq="1min", prominence=0.075, plat_min_abs=0.05):
    data = load.resample(freq).mean()
    peak_idx, *_ = find_peak_plats(data, prominence=prominence, plat_min_abs=plat_min_abs)

    occ_r = occ.resample(freq).max()
    occ_r = occ_r.rolling(2 * 2 + 1, center=True, min_periods=1).max() #margin 2 to the löeft and right
    peak_ts = data.index[peak_idx]
    unoccupied = occ_r.reindex(peak_ts).to_numpy() == 0

    peak_days = pd.Series(peak_ts.date)
    return {
        "peaks_unocc": int(unoccupied.sum()),
        "days_peak_unocc": peak_days[unoccupied].nunique(),
        "share_peaks_unocc": unoccupied.mean() if len(peak_ts) else np.nan,
    }

PARTIAL_CONFIGS = {
    "electricity_hotwater_active": (
        "Electricity Load",
        "Hot Water Load",
        ["Occupancy Home Active"],
    ),
    "electricity_hotwater_inactive": (
        "Electricity Load",
        "Hot Water Load",
        ["Occupancy Home Not Active"],
    ),
    "heating_active_T": (
        "Heating Load",
        "Occupancy Home Active",
        ["T"],
    ),
    "heating_inactive_T": (
        "Heating Load",
        "Occupancy Home Not Active",
        ["T"],
    ),
    "heating_T_inactive": (
        "Heating Load",
        "T",
        ["Occupancy Home Not Active"],
    ),
    "heating_electricity_all": (
        "Heating Load",
        "Electricity Load",
        [
            "T",
            "Occupancy Home Not Active",
            "Occupancy Home Active",
        ],
    ),
}

def prepare_corr_data(data, freq="1min", resid=False, Offday=2):
    data_corr = data.copy()

    if Offday == 0:
        data_corr = data_corr[data_corr["Off Day"] == 1]
    elif Offday == 1:
        data_corr = data_corr[data_corr["Off Day"] == 0]
    elif Offday != 2:
        raise ValueError("Offday must be 0, 1 or 2")

    agg = {
        "Electricity Load": "mean",
        "Hot Water Load": "mean",
        "Heating Load": "mean",
        "Occupancy Home Active": "first",
        "Occupancy Home Not Active": "first",
        "T": "mean",
    }

    data_corr = data_corr.resample(freq).agg(agg)

    if resid:
        tod = data_corr.index.hour * 60 + data_corr.index.minute
        daily_mean = data_corr.groupby(tod).transform("mean")
        data_corr = data_corr - daily_mean

    return data_corr

def corr_table(data,freq="1min",method="spearman",resid=False,Offday=2,partial=True):

    data_corr = prepare_corr_data(data, freq, resid, Offday)

    if resid:
        tod = data_corr.index.hour * 60 + data_corr.index.minute
        mean_dailys = data_corr.groupby(tod).transform("mean")
        data_corr = data_corr - mean_dailys

    corr = data_corr.corr(method=method)

    if not partial:
        return corr, pd.DataFrame()

    partial_results = []
    for name, (x, y, covar) in PARTIAL_CONFIGS.items():
        result = pg.partial_corr(data=data_corr,x=x,y=y,covar=covar,method=method)
        result["comparison"] = name
        result["x"] = x
        result["y"] = y
        result["covar"] = ", ".join(covar)

        partial_results.append(result.reset_index(drop=True))

    partial_df = pd.concat(partial_results, ignore_index=True)
    partial_df = partial_df.set_index("comparison")

    return corr, partial_df

from statsmodels.tsa.stattools import acf
def autocorr_table(data,freq="15min",max_lag="48h",resid=False,Offday=2):
    df = prepare_corr_data(data, freq, resid, Offday)
    step = pd.to_timedelta(freq)
    max_steps = int(pd.to_timedelta(max_lag) / step)
    results = []
    for variable in df.columns:
        series = df[variable]
        acf_values = acf(series,nlags=max_steps,fft=True,missing="conservative")

        results.append(
            pd.DataFrame({
                "variable": variable,
                "lag_min": np.arange(len(acf_values))
                           * step.total_seconds() / 60,
                "r": acf_values,
            })
        )

    return pd.concat(results,ignore_index=True).set_index(["variable", "lag_min"])

def check_duplicate_timeseries(manifest):
    duped_list = manifest["apartment_seeds"].duplicated(keep=False).any()
    exploded = manifest['apartment_seeds'].explode()
    duped_single = exploded[exploded.duplicated(keep=False)].unique()
    print(f"Duplicated Seed Lists: {duped_list}")
    print(f"Duplicated Seeds: {duped_single}")

