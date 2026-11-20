from pathlib import Path
import json
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from skmultilearn.model_selection import IterativeStratification
from scipy.stats import ks_2samp, chi2_contingency
import shutil
import hashlib, json
import os


SPLIT_NAMES = ("train", "val", "test", "holdout_building", "holdout_year", "holdout_station")

def load_split_manifests(paths, validate=True):
    out = {}
    for name, p in paths.items():
        p = Path(p)
        df = pd.read_csv(p, index_col=0) if p.suffix == ".csv" else pd.read_parquet(p)
        out[name] = df.reset_index(drop=True)
    if validate:
        return validate_split_manifests(out)
    return out

def validate_split_manifests(split_dfs):
    missing = {"train", "val", "test"} - set(split_dfs)
    assert not missing, f"missing required splits: {missing}"
    unknown = set(split_dfs) - set(SPLIT_NAMES)
    assert not unknown, f"unknown split names: {unknown}"
    all_runs = pd.concat([d["run_id"] for d in split_dfs.values()])
    assert not all_runs.duplicated().any(), "run_ids occur in more than one split"
    b = {n: set(split_dfs[n]["building_id"]) for n in ("train", "val", "test")}
    for a, c in (("train", "val"), ("train", "test"), ("val", "test")):
        assert not (b[a] & b[c]), f"{len(b[a] & b[c])} buildings in both {a} and {c}"

    holdout_stations = split_dfs["holdout_station"]["weatherFilepath"].str.extract(r"TRY20\d\d_(\d+)_")[0]
    for k, man in split_dfs.items():
        if k != "holdout_station":
            man["station"] = man["weatherFilepath"].str.extract(r"TRY20\d\d_(\d+)_")[0]
            leaks = man[man["station"].isin(holdout_stations)]
            assert leaks.empty, "holdout stations found split"
        req_cols = {"weatherFilepath", "appliances", "weather_seed", "app_ownership_seed"}

        assert not (req_cols - set(man.columns)), "manifest lacks req cols"

        n_weather = man.groupby("weather_seed")["weatherFilepath"].nunique()
        assert not (n_weather > 1).any(), "differing weatherfile despite same seed"

        n_apliances = man.groupby("app_ownership_seed")["appliances"].nunique()
        assert not (n_apliances > 1).any(), "differing appliances despite same seed"




    for n, d in split_dfs.items():
        print(f"{n}: {len(d)} rows, {d['building_id'].nunique()} buildings")
    return split_dfs




def datasplit(X,Y, fractions, seed=None):
    assert abs(sum(fractions) - 1.0) < 1e-6
    if seed is not None:
        np.random.seed(seed)
    indices = np.arange(len(X))
    remaining_X = X
    remaining_Y = Y
    remaining_idx = indices
    remaining_data = 1.0
    splits = []

    for fraction in fractions[:-1]:
        calc_frac = fraction/remaining_data
        stratifier = IterativeStratification(n_splits=2,order=2, sample_distribution_per_fold=[calc_frac, 1-calc_frac])
        keep_idx, take_idx = next(stratifier.split(remaining_X, remaining_Y))
        splits.append(remaining_idx[take_idx])
        remaining_X, remaining_Y = remaining_X[keep_idx], remaining_Y[keep_idx]
        remaining_idx = remaining_idx[keep_idx]
        remaining_data -= fraction

    splits.append(remaining_idx)
    return splits

def split_manifests(manifest, strat_cols,holdout_type="TH", holdout_age="DE.05", holdout_year=None ,fractions=[0.7, 0.15, 0.15], seed=42):

    is_holdout_buildings = (manifest["buildingType"] == holdout_type) & (manifest["buildingAgeBin"] == holdout_age)

    if holdout_year is None:
        is_holdout_year = pd.Series(False, index=manifest.index)
    else:
        is_holdout_year = manifest["year"] == holdout_year

    strat_split_pool = manifest[~is_holdout_buildings & ~is_holdout_year]

    buildings = strat_split_pool.drop_duplicates("building_id").set_index("building_id")

    holdout_buildings = manifest.loc[is_holdout_buildings, "building_id"].unique()

    strat_df = buildings[strat_cols].astype(str).copy()
    Y = pd.get_dummies(strat_df).values
    X = np.arange(len(buildings)).reshape(-1, 1)

    train_idx, val_idx, test_idx = datasplit(X, Y, fractions=fractions, seed=seed)

    train_buildings = buildings.index[train_idx]
    val_buildings = buildings.index[val_idx]
    test_buildings = buildings.index[test_idx]

    train_df = strat_split_pool[strat_split_pool['building_id'].isin(train_buildings)]
    val_df = strat_split_pool[strat_split_pool['building_id'].isin(val_buildings)]
    test_df = strat_split_pool[strat_split_pool['building_id'].isin(test_buildings)]

    holdout_building_df = manifest[is_holdout_buildings & ~is_holdout_year]
    holdout_year_df = manifest[~is_holdout_buildings & is_holdout_year]
    holdout_both_df = manifest[is_holdout_buildings & is_holdout_year]

    print(f"buildings -> train {len(train_buildings)}, val {len(val_buildings)}, test {len(test_buildings)}")
    print(f"rows-> train {len(train_df)}, val {len(val_df)}, test {len(test_df)}")
    print(f"rows-> holdout_building {len(holdout_building_df)}, holdout_year {len(holdout_year_df)}, holdout_both {len(holdout_both_df)}")

    return buildings,train_idx, val_idx, test_idx, train_buildings,val_buildings,test_buildings ,train_df, val_df, test_df, holdout_buildings, holdout_building_df, holdout_year_df, holdout_both_df


def check_distr(split_dfs,cont_cols, cat_cols, row_cols):
    building_dfs = {n: d.drop_duplicates("building_id") for n, d in split_dfs.items()}
    POOL = ("train", "val", "test")
    HOLDOUTS = ("holdout_building", "holdout_year", "holdout_station")
    orig_buildings = pd.concat([building_dfs[n] for n in POOL])

    def annotate(ax, lines):
        ax.text(0.98, 0.98, "\n".join(lines), transform=ax.transAxes, ha="right", va="top",
                fontsize=9, family="monospace",bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))


    for col in cont_cols:
        print(f"___{col}___")
        summary = pd.DataFrame({
            "original": orig_buildings[col].describe(percentiles=[.25, .5, .75]),
            **{name: df[col].describe(percentiles=[.25, .5, .75]) for name, df in building_dfs.items()}

        })
        print(summary.round(2))

        fig, ax = plt.subplots(figsize=(6, 4))
        ks_lines = []

        for name, df in building_dfs.items():

            if name in HOLDOUTS:
                continue
            stat, p = ks_2samp(orig_buildings[col], df[col])  # KS vs. pooled original
            print(f"{name}: stat={stat:.4f}, p={p:.4f}")
            ks_lines.append(f"KS {name:<5} p={p:.3f}")
            ax.hist(df[col], bins=30, density=True, histtype="step", label=name)
        ax.set_title(col)
        ax.legend(loc="upper left")
        annotate(ax, ks_lines)
        plt.tight_layout()
        plt.show()


    cat_props = {}
    for col in cat_cols:
        print(f"___{col}___")
        summary = pd.DataFrame({
            "original": orig_buildings[col].value_counts(normalize=True),
            **{name: df[col].value_counts(normalize=True) for name, df in building_dfs.items()}
        })
        print(summary.round(2))
        cat_props[col] = summary[["original", *POOL]].fillna(0)

        conting_tab = pd.DataFrame({n: building_dfs[n][col].value_counts() for n in POOL}).fillna(0)

        chi2, p, dof, _ = chi2_contingency(conting_tab)
        print(f"Chi2={chi2:.4f}, p={p:.4f}, dof={dof}")

        fig, ax = plt.subplots(figsize=(6, 4))
        tab = summary[["original", *POOL]].fillna(0).sort_index()
        tab.columns = [f"{c} (n={len(orig_buildings) if c == 'original' else len(building_dfs[c])})"for c in tab.columns]
        tab.plot.bar(ax=ax, width=0.8)
        ax.set_title(col)
        ax.set_ylabel("share")
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=45)
        annotate(ax, [f"chi2 p={p:.3f} (dof={dof})"])
        plt.tight_layout()
        plt.show()

    row_props = {}
    for col in row_cols:
        print(f"___{col} (rows)___")

        summary = pd.DataFrame({n: d[col].value_counts(normalize=True) for n, d in split_dfs.items()})
        print(summary.round(3))
        chi2, p, dof, _ = chi2_contingency(pd.DataFrame({n: split_dfs[n][col].value_counts() for n in POOL}).fillna(0))
        row_props[col] = summary[list(POOL)].fillna(0)
        print(f"Chi2={chi2:.4f}, p={p:.4f}, dof={dof}")

        fig, ax = plt.subplots(figsize=(6, 4))
        summary[list(POOL)].fillna(0).sort_index().plot.bar(ax=ax, width=0.8)
        ax.set_title(f"{col} (rows)")
        ax.set_ylabel("share")
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=45)
        annotate(ax, [f"chi2 p={p:.3f} (dof={dof})"])
        plt.tight_layout()
        plt.show()




def move_split_data( manifest,source_dir, split_dirs , split_dfs):
    for d in split_dirs.values():
                d.mkdir(parents=True, exist_ok=True)

    split_by_run_id = {}

    for split, df in split_dfs.items():
        for run_id in df["run_id"]:
            assert run_id not in split_by_run_id, f"{run_id} is in two splits"
            split_by_run_id[run_id] = split

    assert manifest["run_id"].isin(split_by_run_id).all(), "run missing from splits"

    copied = 0
    missing = []
    for run_id, split in tqdm(split_by_run_id.items(), total=len(split_by_run_id)):
        src = source_dir / f"{run_id}.h5"
        trgt = split_dirs[split] / f"{run_id}.h5"

        if not src.exists():
            missing.append(src)
            continue

        shutil.copy2(src, trgt)
        copied += 1

    print(f"Copied: {copied}\nMissing: {len(missing)}")

def compare_distributions(buildings, train_idx, val_idx, test_idx, cols):
    splits = {'train': buildings.loc[train_idx], 'val': buildings.loc[val_idx], 'test': buildings.loc[test_idx]}

    for col in cols:
        print(f"=== {col} ===")
        orig = buildings[col]

        summary = pd.DataFrame({
            'original': orig.describe(percentiles=[.25, .5, .75]),
            **{name: df[col].describe(percentiles=[.25, .5, .75]) for name, df in splits.items()}
        })
        print(summary.round(2))

        print("\nKS test vs original (p < 0.05 suggests distributions differ):")
        for name, df in splits.items():
            stat, p = ks_2samp(orig, df[col])
            print(f"  {name}: statistic={stat:.4f}, p={p:.4f}")
        print()

def save_stats(path, res, cols, stats, norm_by_col=None):
    if all(isinstance(v, dict) for v in stats.values()):
        names = next(iter(stats.values())).keys()
        stats = {n: np.array([stats[c][n] for c in cols]) for n in names}
    np.savez(Path(path), columns=np.array(cols), resolution=str(res),norm_by_col=json.dumps(norm_by_col or {}, sort_keys=True), **stats)

def load_stats(path, return_meta=False):
    path = Path(path)
    data = np.load(path)
    cols = data["columns"].tolist()


    meta_keys = ("columns", "resolution", "norm_by_col")
    names = [k for k in data.files if k not in meta_keys]
    stats = {col: {name: float(data[name][i]) for name in names} for i, col in enumerate(cols)}

    if not return_meta:
        return stats
    meta = {
            "resolution": str(data["resolution"]),
      # files saved before this change have no norm_by_col entry
        "norm_by_col": json.loads(str(data["norm_by_col"])) if "norm_by_col" in data.files else {},
    }
    return stats, meta

def make_unique_stat_file_name(res, norm_name, norm_by_col, cols, run_ids, dir):
    key = json.dumps({"cols": cols, "norm": norm_by_col, "runs": sorted(run_ids)},
                     sort_keys=True, default=str)
    h = hashlib.sha256(key.encode()).hexdigest()[:8]
    return dir / f"stats_{norm_name}_{res}_{h}.npz"

def load_precalc_stats(res, norm_name, cols, run_ids, norm_variant, dir):
    norm_by_col = norm_variant[norm_name]
    path = make_unique_stat_file_name(res, norm_name, norm_by_col, cols, run_ids, dir)
    stats, meta = load_stats(path, return_meta=True)
    if meta["resolution"] != res or meta["norm_by_col"] != (norm_by_col or {}):
        raise ValueError(f"{path.name}: metadata does not match requested config")
    return stats,meta

def calc_scaler_stats(files,run_ids, columns, aggregation, resolution=None,norm_factors=None, quantiles=(0.01, 0.25, 0.5, 0.75, 0.99), sample_per_file=500, seed=42):
    rng=np.random.default_rng(seed)
    n_cols = len(columns)
    n=0
    total = np.zeros(len(columns),dtype=np.float64)
    total_sq = np.zeros(len(columns),dtype=np.float64)
    log_total = np.zeros(len(columns),dtype=np.float64)
    log_total_sq = np.zeros(len(columns),dtype=np.float64)
    col_min = np.full(n_cols,np.inf)
    col_max = np.full(n_cols,-np.inf)
    samples = []
    for path, run_id in tqdm(list(zip(files, run_ids))):
        if not Path(path).exists():
            print(f"missing:{path}")
            continue
        ts = remove_daylightsaving(pd.read_hdf(path, key="timeseries")[columns])

        if resolution is not None:
            agg = {col: aggregation[col] for col in columns}
            ts = ts.resample(resolution).agg(agg)

        arr = ts.to_numpy(dtype=np.float64)
        if norm_factors:
            arr = arr / norm_factors[run_id].T

        if not np.isfinite(arr).all():
            raise ValueError(f"nan problem")

        total += arr.sum(axis=0)
        total_sq += np.square(arr).sum(axis=0)
        col_min = np.minimum(col_min, arr.min(axis=0))
        col_max = np.maximum(col_max, arr.max(axis=0))


        log_arr = np.log1p(np.maximum(arr, 0))
        log_total += log_arr.sum(axis=0)
        log_total_sq += np.square(log_arr).sum(axis=0)

        n += arr.shape[0]
        samples.append(arr[rng.integers(0, arr.shape[0], size=sample_per_file)])

    mean = total/n
    variance = (total_sq/n - mean**2)
    std = np.sqrt(np.maximum(variance, 1e-12))

    valid_log_cols = col_min >= 0
    log_mean = np.full(n_cols, np.nan)
    log_std = np.full(n_cols, np.nan)
    log_mean[valid_log_cols] = log_total[valid_log_cols] / n
    log_variance = (log_total_sq[valid_log_cols] / n- log_mean[valid_log_cols]**2)
    log_std[valid_log_cols] = np.sqrt(np.maximum(log_variance, 1e-12))

    stats = {"mean": mean, "std": std, "min": col_min, "max": col_max, "log_mean": log_mean, "log_std": log_std}
    pooled = np.concatenate(samples)
    for q, vals in zip(quantiles, np.quantile(pooled, quantiles, axis=0)):
        stats[f"q{q}"] = vals
    return stats

def remove_daylightsaving(ts):
    if ts.index.tz is None:
        return ts

    step = ts.index.to_series().diff().min()
    ts = ts.set_axis(ts.index.tz_localize(None))
    ts = ts[~ts.index.duplicated(keep="first")]

    grid = pd.date_range(ts.index[0], ts.index[-1], freq=step)
    if len(grid) != len(ts):
        ts = ts.reindex(grid).interpolate(method="time", limit_direction="both")
    return ts