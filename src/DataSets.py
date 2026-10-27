from pathlib import Path
import json
import pandas as pd
import ast
import numpy as np
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import random
import torch
import math
from skmultilearn.model_selection import IterativeStratification
from scipy.stats import ks_2samp, chi2_contingency
import shutil
from torch.utils.data import IterableDataset, get_worker_info
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from src.DataSplitting import remove_daylightsaving

def season_channels(index):
    frac = (index.dayofyear - 1 + (index.hour * 60 + index.minute) / 1440) / 365.25
    ang = 2 * np.pi * frac.to_numpy()
    return np.stack([np.sin(ang), np.cos(ang)]).astype(np.float32)

class h5Dataset(IterableDataset):
    def __init__(self,files,rund_ids,gen_cols,cond_cols,window_length,stride,target_scaler,condition_scaler, target_norm,condition_norm,
                 static_by_run_id,aggregation,resolution=None,shufflefiles=False,
                 shufflewindows=True, buffer_size=0,context_steps=0, context_dropout=0.0, context_noise=0.0, use_season=False,context_reverse=False):
        super().__init__()
        self.files = list(files)
        self.run_ids = list(rund_ids)
        self.gen_cols = list(gen_cols)
        self.cond_cols = list(cond_cols)
        self.all_cols = list(dict.fromkeys(self.gen_cols + self.cond_cols))
        self.window_length = window_length
        self.stride = stride
        self.target_scaler = target_scaler
        self.condition_scaler = condition_scaler
        self.static_by_run_id = static_by_run_id
        self.resolution = resolution
        self.shufflefiles = shufflefiles
        self.shufflewindows = shufflewindows
        self.buffer_size = buffer_size
        self.aggregation = aggregation
        self.target_norm = target_norm
        self.condition_norm = condition_norm
        self.context_steps = context_steps
        self.context_dropout = context_dropout
        self.context_noise = context_noise
        self.use_season = use_season
        self.context_reverse = context_reverse

        if len(self.files) != len(self.run_ids):
            raise ValueError("input doesnt have same length")

        missing_files = [path for path in self.files if not Path(path).exists()]
        missing_norm = [r for r in self.run_ids if r not in self.target_norm or r not in self.condition_norm]
        if missing_norm:
            raise KeyError(f"{len(missing_norm)} run_ids have no norm factors {missing_norm[:3]}")
        if missing_files:
            raise FileNotFoundError(f"{len(missing_files)} files are missing")


    def _read_series(self,path):
        ts = remove_daylightsaving(pd.read_hdf(path, key="timeseries", columns=self.all_cols))

        if self.resolution is not None:
            agg= {col: self.aggregation[col] for col in self.all_cols}
            ts = ts.resample(self.resolution).agg(agg)

        target = ts[self.gen_cols].to_numpy(dtype=np.float32, copy=False).T
        condition = ts[self.cond_cols].to_numpy(dtype=np.float32, copy=False).T
        season = season_channels(ts.index) if self.use_season else None
        return target, condition,season

    def _add_context(self, target, condition_window, start):
        K = self.context_steps
        if K == 0:
            return condition_window
        n = target.shape[0]
        ctx = np.zeros((n + 1, self.window_length), dtype=np.float32)  # n target channels + 1 mask
        if start >= K and random.random() >= self.context_dropout:
            tail = target[:, start - K:start]  # clean, already scaled
            if self.context_noise > 0:
                tail = tail + self.context_noise * np.random.randn(*tail.shape).astype(np.float32)
            if self.context_reverse:
                tail = tail[:, ::-1]
            ctx[:n, :K] = tail
            ctx[n, :K] = 1.0
        return np.concatenate([condition_window, ctx], axis=0)

    def __iter__(self):
        workers = get_worker_info()
        items = list(zip(self.files, self.run_ids))

        if workers is not None:
            items = items[workers.id::workers.num_workers]

        if self.shufflefiles:
            random.shuffle(items)

        buffer = []

        for path, run_id in items:
            target, condition, season = self._read_series(path)
            target = target / self.target_norm[run_id]
            condition = condition / self.condition_norm[run_id]

            # normalisierung basierend auf gewähltem scaler
            target = apply_scaler(target, self.target_scaler)
            condition = apply_scaler(condition, self.condition_scaler)

            starts = list(range(0, target.shape[1] - self.window_length+ 1, self.stride))

            if self.shufflewindows:
                random.shuffle(starts)

            static = self.static_by_run_id[run_id]

            for start in starts:
                end = start + self.window_length

                target_window = np.ascontiguousarray(target[:,start:end])
                condition_window = np.ascontiguousarray(condition[:,start:end])
                if season is not None:
                    condition_window = np.concatenate([condition_window, season[:, start:end]], axis=0)
                condition_window = self._add_context(target, condition_window, start)

                sample = {
                    "target": torch.from_numpy(target_window),
                    "condition": torch.from_numpy(condition_window),
                    "static_cat": static["categorical"],
                    "static_cont": static["continuous"],
                    "static_bool": static["boolean"],
                }


                sample["target_norm"] = torch.from_numpy(self.target_norm[run_id])
                sample["condition_norm"] = torch.from_numpy(self.condition_norm[run_id])

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

class SampleWindowDataset(h5Dataset):
    def __init__(self, index, split_files, **kwargs):

        path_by_run = {Path(p).stem: Path(p) for p in split_files}
        available = index["run_id"].isin(path_by_run)

        if not available.all():
            missing = index.loc[~available, "run_id"].nunique()
            print(f"skipping {missing} runs without a local file "
                  f"({(~available).sum()} windows)")

        index = index[available].reset_index(drop=True)

        run_ids = list(dict.fromkeys(index["run_id"]))

        files = [path_by_run[r] for r in run_ids]

        super().__init__(files,run_ids,shufflefiles=False, shufflewindows=False,buffer_size=0, **kwargs)
        self.index = index.reset_index(drop=True)
        self.starts_per_run = {rid: g for rid,g in self.index.groupby("run_id")}

    def __len__(self):
        return len(self.index)

    def __iter__(self):
        items =list(zip(self.files, self.run_ids))
        w = get_worker_info()
        if w is not None:
            items = items[w.id::w.num_workers]

        for path, run_id in items:
            target, condition, season = self._read_series(path)
            if self.target_norm is not None:
                target = target/ self.target_norm[run_id]
                condition = condition/self.condition_norm[run_id]
            target = apply_scaler(target, self.target_scaler)
            condition = apply_scaler(condition, self.condition_scaler)

            static = self.static_by_run_id[run_id]

            for row, start in zip(self.starts_per_run[run_id].index,self.starts_per_run[run_id]["start_hour"]):
                end = start + self.window_length
                assert end <= target.shape[1], f"{run_id}: window {start}-{end} out of range"

                target_window = np.ascontiguousarray(target[:,start:end])
                condition_window = np.ascontiguousarray(condition[:,start:end])
                if season is not None:
                    condition_window = np.concatenate([condition_window, season[:, start:end]], axis=0)
                condition_window = self._add_context(target, condition_window, start)

                sample = {
                    "row": row,
                    "target": torch.from_numpy(target_window),
                    "condition": torch.from_numpy(condition_window),
                    "static_cat": static["categorical"],
                    "static_cont": static["continuous"],
                    "static_bool": static["boolean"],
                }

                yield sample



def draw_sampling_windows(split_df,strat_cols, n_runs, n_windows, window_steps, seed=42):
    rng = np.random.default_rng(seed)
    # select run
    grps = split_df.groupby(strat_cols)
    n = math.ceil((n_runs/len(grps)))
    assert n > 0, f"n_runs must be chosen higher"
    picked_rows = []
    for key, grp in grps:
        row = grp.sample(n=n,random_state=int(rng.integers(1<<31)))
        picked_rows.append(row)

    rows = pd.concat(picked_rows)

    #select days
    month_of_day = pd.DataFrame(pd.date_range("2026-01-01", periods=365).month, columns=["month"])
    output = []
    for _,r in rows.iterrows():
        day_windows = month_of_day.groupby(month_of_day["month"]).apply(lambda s: s.sample(1, random_state=int(rng.integers(1 << 31))))
        # get actual indices since daywindows are tuples cause the multiindex of groupby
        day_idx = day_windows.index.get_level_values(1).to_numpy()
        additional = n_windows - len(day_idx)
        # minimum of 12 Windows no matter the input param
        if additional > 0:
            rest = np.setdiff1d(np.arange(365), day_idx)
            day_idx_new = rng.choice(rest, additional, replace=False)
            day_idx = np.concatenate([day_idx, day_idx_new])
        for d in np.sort(day_idx):
            output.append((
                r["run_id"],
                int(d) * window_steps,
                int(d),
                r["building_id"],
                r["buildingType"],
                r["buildingAgeBin"],
                r["year"]

            ))

    return pd.DataFrame(output,columns=["run_id", "start_hour", "day", "building_id",
                                       "buildingType", "buildingAgeBin", "year"])

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

def select_scalers(stats, columns, scaler_by_col, default="zscore", zero_cols=()):
    center, scale, log_mask = [], [], []
    for col in columns:
        s = stats[col]
        kind = scaler_by_col.get(col, default)
        if kind == "zscore":
            c, sc, lg = s["mean"], s["std"], False
        elif kind == "log_zscore":
            c, sc, lg = s["log_mean"], s["log_std"], True
        elif kind == "minmax":
            c, sc, lg = s["min"], s["max"] - s["min"], False
        elif kind == "robust":
            c, sc, lg = s["q0.5"], s["q0.75"] - s["q0.25"], False
        else:
            raise ValueError(f"unknown scaler {kind!r} for {col}")
        if np.isnan(c) or np.isnan(sc):
            raise ValueError(f"{kind} stats are NaN for {col} (log needs a non-negative column)")
        center.append(0.0 if col in zero_cols else c)
        scale.append(sc)
        log_mask.append(lg)
    to_t = lambda v: torch.tensor(v, dtype=torch.float32).unsqueeze(1)
    return to_t(center), to_t(scale), torch.tensor(log_mask, dtype=torch.bool)


def apply_scaler(x, scaler):
    center, scale, log_mask = (t.numpy() for t in scaler)
    x = x.copy()
    x[log_mask] = np.log1p(np.maximum(x[log_mask], 0))
    return (x - center) / (scale + 1e-8)


def invert_scaler(x, scaler, device=None):
    center, scale, log_mask = (t.to(x.device) for t in scaler)
    x = x * (scale + 1e-8) + center
    x[:, log_mask] = torch.expm1(x[:, log_mask])
    return x

def build_norm_factors(manifest_split, columns, norm_by_col):
    factors = {}
    for row in manifest_split.itertuples(index=False):
        f = np.ones((len(columns), 1), dtype=np.float32)
        for i, col in enumerate(columns):
            key = norm_by_col.get(col)
            if key is not None:
                f[i, 0] = getattr(row, key)
        factors[row.run_id] = f
    return factors

def build_condition(preprocessor, train_df, params, batch_size, device="cpu"):
    param_dict = {}
    for col in preprocessor.static_cat_cols:
        param_dict[col] = train_df[col].mode().iloc[0]
    for col in preprocessor.static_cont_cols:
        param_dict[col] = train_df[col].median()
    for col in preprocessor.static_bool_cols:
        param_dict[col] = bool(train_df[col].mode().iloc[0])

    param_dict.update(params)

    df = pd.DataFrame([param_dict] * batch_size)
    out = preprocessor.transform(df)

    return {
        "static_cat": torch.from_numpy(out["categorical"]).long().to(device),
        "static_cont": torch.from_numpy(out["continuous"]).float().to(device),
        "static_bool": torch.from_numpy(out["boolean"]).float().to(device),
    }
