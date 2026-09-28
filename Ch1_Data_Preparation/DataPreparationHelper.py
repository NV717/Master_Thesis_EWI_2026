from pathlib import Path
import json
import pandas as pd
import ast
import numpy as np
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import random
import torch
from skmultilearn.model_selection import IterativeStratification
from scipy.stats import ks_2samp
import shutil
from pathlib import Path

RESULTS_DIR = r"DataPreparation/Generation/Generation"
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


# ddataset
from torch.utils.data import IterableDataset, get_worker_info
class h5Dataset(IterableDataset):
    def __init__(self,files,rund_ids,gen_cols,cond_cols,window_length,stride,target_mean,target_std,condition_mean,condition_std,static_by_run_id,resolution=None,shufflefiles=False,shufflewindows=True, buffer_size=0):
        super().__init__()
        self.files = list(files)
        self.run_ids = list(rund_ids)
        self.gen_cols = list(gen_cols)
        self.cond_cols = list(cond_cols)
        self.all_cols = list(dict.fromkeys(self.gen_cols + self.cond_cols))
        self.window_length = window_length
        self.stride = stride
        self.target_mean = target_mean.numpy() if torch.is_tensor(target_mean) else target_mean
        self.target_std = target_std.numpy() if torch.is_tensor(target_std) else target_std
        self.condition_mean = condition_mean.numpy() if torch.is_tensor(condition_mean) else condition_mean
        self.condition_std = condition_std.numpy() if torch.is_tensor(condition_std) else condition_std
        self.static_by_run_id = static_by_run_id
        self.resolution = resolution
        self.shufflefiles = shufflefiles
        self.shufflewindows = shufflewindows
        self.buffer_size = buffer_size

        if len(self.files) != len(self.run_ids):
            raise ValueError("input doesnt have same length")

        missing_files = [path for path in self.files if not Path(path).exists()]

        if missing_files:
            raise FileNotFoundError(f"{len(missing_files)} files are missing")


    def _read_series(self,path):
        ts = pd.read_hdf(path, key="timeseries", columns=self.all_cols)

        if self.resolution is not None:
            ts = ts.resample(self.resolution).mean()

        target = ts[self.gen_cols].to_numpy(dtype=np.float32, copy=False).T
        condition = ts[self.cond_cols].to_numpy(dtype=np.float32, copy=False).T

        return target, condition

    def __iter__(self):
        workers = get_worker_info()
        items = list(zip(self.files, self.run_ids))

        if workers is not None:
            items = items[workers.id::workers.num_workers]

        if self.shufflefiles:
            random.shuffle(items)

        buffer = []

        for path, run_id in items:
            target, condition = self._read_series(path)

            target = (target-self.target_mean) / (self.target_std+1e-8)
            condition = (condition-self.condition_mean) / (self.condition_std+1e-8)
            starts = list(range(0, target.shape[1] - self.window_length+1, self.stride))
            if self.shufflewindows:
                random.shuffle(starts)

            static = self.static_by_run_id[run_id]

            for start in starts:
                end = start + self.window_length

                target_window = np.ascontiguousarray(target[:,start:end])
                condition_window = np.ascontiguousarray(condition[:,start:end])

                sample = {
                    "target": torch.from_numpy(target_window),
                    "condition": torch.from_numpy(condition_window),
                    "static_cat": static["categorical"],
                    "static_cont": static["continuous"],
                    "static_bool": static["boolean"],
                }

                if self.buffer_size <= 0:
                    yield sample
                    continue

                if len(buffer) < self.buffer_size:
                    buffer.append(sample)
                else:
                    idx = random.randrange(self.buffer_size)
                    sample_to_yield = buffer[idx]
                    buffer[idx] = sample
                    yield sample_to_yield

        random.shuffle(buffer)
        for sample in buffer:
            yield sample

from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

class PreprocessStaticCols:
    def __init__(self, static_cat_cols, static_cont_cols, static_bool_cols):
        self.cat_encoder = OrdinalEncoder(handle_unknown="use_encoded_value",unknown_value=-1)
        self.cont_scaler = StandardScaler()
        self.static_cat_cols = static_cat_cols
        self.static_cont_cols = static_cont_cols
        self.static_bool_cols = static_bool_cols

    def fit(self, train_manifest):
        self.cat_encoder.fit(train_manifest[self.static_cat_cols])
        self.cont_scaler.fit(train_manifest[self.static_cont_cols])
        return self

    def transform(self, manifest):
        cat = self.cat_encoder.transform(manifest[self.static_cat_cols]).astype(np.int64)
        cat += 1

        cont = self.cont_scaler.transform(manifest[self.static_cont_cols]).astype(np.float32)

        boolean = manifest[self.static_bool_cols].astype(np.float32).to_numpy(copy=True)

        return {
            "categorical": cat,
            "continuous": cont,
            "boolean": boolean,
        }

def existing_filter(manifest_split, split_dir):
    split_dir = Path(split_dir)
    exist = manifest_split["run_id"].apply(lambda run_id:(split_dir / f"{run_id}.h5").exists())
    missing = manifest_split.loc[~exist, "run_id"].tolist()
    filtered = manifest_split.loc[exist].reset_index(drop=True)
    print(f"Missing Files: {len(missing)}\nExisting Files: {len(filtered)}\nTotal: {len(exist)}")
    return filtered, missing

def normalizestatic(manifest_split, split_dir, preprocessor):
    run_ids = manifest_split["run_id"].tolist()
    files = [Path(split_dir) / f"{run_id}.h5" for run_id in run_ids]
    transformed = preprocessor.transform(manifest_split)

    static_by_run_id = {}

    for i, run_id in enumerate(run_ids):
        static_by_run_id[run_id] = {
             "categorical": torch.from_numpy(transformed["categorical"][i]).long(),
            "continuous": torch.from_numpy(transformed["continuous"][i]).float(), # ToDo check impact on stuff like n_persons since its scaled as continous
            "boolean": torch.from_numpy(transformed["boolean"][i]).float(),
        }
    return files, run_ids, static_by_run_id


def stepsperday(resolution):
    if resolution is None:
        return 24*60
    return int(pd.Timedelta("1D") /pd.Timedelta(resolution))

# todo make same thing for min max scaler
def calc_mean_std(files, columns, resolution=None):
    n=0
    total = np.zeros(len(columns),dtype=np.float64)
    total_sq = np.zeros(len(columns),dtype=np.float64)

    for path in files:
        if not Path(path).exists():
            print(f"missing:{path}")
            continue
        ts = pd.read_hdf(path, key="timeseries")[columns]

        if resolution is not None:
            ts = ts.resample(resolution).mean()

        arr = ts.to_numpy(dtype=np.float64)

        if not np.isfinite(arr).all():
            raise ValueError(f"nan problem")

        total += arr.sum(axis=0)
        total_sq += np.square(arr).sum(axis=0)
        n += arr.shape[0]

    mean = total/n
    variance = (total_sq/n - mean**2)
    std = np.sqrt(np.maximum(variance, 1e-12))

    return torch.from_numpy(mean).float().unsqueeze(1), torch.from_numpy(std).float().unsqueeze(1)

def datasplit(X,Y, fractions):
    assert abs(sum(fractions) - 1.0) < 1e-6
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

def split_manifests(manifest, strat_cols,holdout_type="TH", holdout_age="DE.05", fractions=[0.7, 0.15, 0.15]):
    buildings = manifest.drop_duplicates("building_id").set_index("building_id")

    holdout_buildings = buildings.index[
        (buildings["buildingType"] == holdout_type) & (buildings["buildingAgeBin"] == holdout_age)]
    buildings = buildings.loc[
        ~((buildings["buildingType"] == holdout_type) & (buildings["buildingAgeBin"] == holdout_age))]

    strat_df = buildings[strat_cols].astype(str).copy()
    Y = pd.get_dummies(strat_df).values
    X = np.arange(len(buildings)).reshape(-1, 1)

    train_idx, val_idx, test_idx = datasplit(X, Y, fractions=fractions)

    train_buildings = buildings.index[train_idx]
    val_buildings = buildings.index[val_idx]
    test_buildings = buildings.index[test_idx]

    train_df = manifest[manifest['building_id'].isin(train_buildings)]
    val_df = manifest[manifest['building_id'].isin(val_buildings)]
    test_df = manifest[manifest['building_id'].isin(test_buildings)]

    print(f"buildings -> train {len(train_buildings)}, val {len(val_buildings)}, test {len(test_buildings)}")
    print(f"rows-> train {len(train_df)}, val {len(val_df)}, test {len(test_df)}")

    return buildings,train_idx, val_idx, test_idx, train_buildings,val_buildings,test_buildings ,train_df, val_df, test_df, holdout_buildings


def check_distr(buildings, train_buildings, val_buildings, test_buildings, strat_cols_test,rest_cols):
    splits = {'train': buildings.loc[train_buildings], 'val': buildings.loc[val_buildings], 'test': buildings.loc[test_buildings]}

    for col in rest_cols:
        print(f"=== {col} ===")
        orig = buildings[col]

        summary = pd.DataFrame({
            'original': orig.describe(percentiles=[.25, .5, .75]),
            **{name: df[col].describe(percentiles=[.25, .5, .75]) for name, df in splits.items()}
        })
        print(summary.round(2))
        # todo chi quadrat für kategorische
        print("\nKS test (p < 0.05 suggests distributions differ):")
        for name, df in splits.items():
            stat, p = ks_2samp(orig, df[col])
            print(f"{name}: stat={stat:.4f}, p={p:.4f}")
        print()

    for col in strat_cols_test:
        print(col)
        print(pd.concat({
            "original": buildings.loc[:, col].value_counts(normalize=True),
            'train': buildings.loc[train_buildings, col].value_counts(normalize=True),
            'val': buildings.loc[val_buildings, col].value_counts(normalize=True),
            'test': buildings.loc[test_buildings, col].value_counts(normalize=True),
        }, axis=1).round(3))
        print()

    fig, axes = plt.subplots(1, len(rest_cols), figsize=(6 * len(rest_cols), 4))
    splits = {'original': buildings, 'train': buildings.loc[train_buildings],
              'val': buildings.loc[val_buildings], 'test': buildings.loc[test_buildings]}

    for ax, col in zip(axes, rest_cols):
        for name, df in splits.items():
            ax.hist(df[col], bins=30, density=True, histtype='step', label=name)
        ax.set_title(col)
        ax.legend()

    plt.tight_layout()
    plt.show()

def move_split_data( manifest,source_dir, split_dirs ,train_buildings, val_buildings, test_buildings, holdout_buildings):
    for d in split_dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    split_by_building = {}
    for bid in train_buildings:
        split_by_building[bid] = "train"
    for bid in val_buildings:
        split_by_building[bid] = "val"
    for bid in test_buildings:
        split_by_building[bid] = "test"
    for bid in holdout_buildings:
        split_by_building[bid] = "holdout_test"

    assert manifest["building_id"].isin(split_by_building).all(), "building missing"

    copied, missing = 0, []
    for _, row in tqdm(manifest.iterrows(), total=len(manifest)):
        split = split_by_building[row["building_id"]]
        src = source_dir / f"{row["run_id"]}.h5"
        trgt = split_dirs[split] / f"{row["run_id"]}.h5"

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


def save_stats(path, res, cols, mean, std):
    path = Path(path)
    np.savez(path, columns=np.array(cols), resolution=str(res),mean=mean.squeeze(1).cpu().numpy(),std=std.squeeze(1).cpu().numpy())

def load_stats(path):
    path = Path(path)
    data = np.load(path)
    cols = data["columns"].tolist()
    mean = data["mean"]
    std = data["std"]

    stats = {
        col: {
            "mean": float(mean[i]),
            "std": float(std[i]),
        }
        for i, col in enumerate(cols)
    }

    return stats

def select_stats(stats, columns):
    mean = torch.tensor([stats[col]["mean"] for col in columns], dtype=torch.float32).unsqueeze(1)
    std = torch.tensor([stats[col]["std"] for col in columns],dtype=torch.float32).unsqueeze(1)
    return mean, std