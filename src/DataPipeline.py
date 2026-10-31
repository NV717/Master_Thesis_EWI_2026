from __future__ import annotations
import hashlib
import json
from pathlib import Path
import joblib
from src.DataConfig import DataConfig
from src.DataSets import PreprocessStaticCols, existing_filter, select_scalers, scaled_minmax, build_norm_factors, h5Dataset, SampleWindowDataset, normalizestatic, draw_sampling_windows,build_condition
from src.DataSplitting import split_manifests, load_precalc_stats, calc_scaler_stats, save_stats, make_unique_stat_file_name
from torch.utils.data import DataLoader



def make_splits(manifest, strat_cols,split_params):
    out = split_manifests(manifest, strat_cols,**split_params)
    return {"train": out[7], "val": out[8], "test": out[9],
            "holdout_building": out[11], "holdout_year": out[12], "holdout_both": out[13]}

def add_share_cols(train_df, prefix="apl_"):
    cols = [c for c in train_df.columns if c.startswith(prefix) and train_df[c].std() > 0]
    return list(train_df[cols].T.drop_duplicates().index)

def hash_checker(split_dfs):
    ids = {k: sorted(df["run_id"].tolist()) for k, df in split_dfs.items()}
    return hashlib.sha256(json.dumps(ids, sort_keys=True).encode()).hexdigest()[:12]

class DataPipeline:
    def __init__(self, cfg, split_dfs, preprocessor, stats, split_hash_, missing):
        self.cfg = cfg
        self.split_dfs = split_dfs # after existing filter
        self.preprocessor = preprocessor
        self.stats = stats
        self.split_hash = split_hash_ # before filter
        self.missing = missing
        self.target_scaler = select_scalers(stats, cfg.target_cols, scaler_by_col=cfg.scaler_by_col, zero_cols=cfg.zero_cols)
        self.condition_scaler = select_scalers(stats, cfg.dynamic_cond_cols, scaler_by_col=cfg.scaler_by_col, zero_cols=cfg.zero_cols)

    @classmethod
    def build(cls, cfg, manifest, compute_stats_if_missing=False):
        full= make_splits(manifest, cfg.strat_cols, cfg.split_params)
        h = hash_checker(full)
        filtered = {}
        missing = {}
        for name, df in full.items():
            filtered[name], missing[name] = existing_filter(df, cfg.split_dirs[name])
        preprocessor = PreprocessStaticCols(static_cat_cols=cfg.static_cat_cols,static_cont_cols=cfg.static_cont_cols,
                                            static_bool_cols=cfg.static_bool_cols).fit(filtered["train"])
        stats = cls.load_scaler_stats(cfg, filtered["train"], compute_stats_if_missing)
        return cls(cfg, filtered, preprocessor, stats, h, missing)

    @staticmethod
    def load_scaler_stats(cfg,train_df, compute):
        run_ids = train_df["run_id"].to_list()
        stats_dir = Path(cfg.stats_dir)
        try:
            stats, meta = load_precalc_stats(cfg.resolution, cfg.norm_method,cfg.stat_calc_columns, run_ids, cfg.norm_variant,stats_dir)
            return stats
        except FileNotFoundError:
            if not compute:
                raise
        print("no files found, computing stats insted")
        files = [Path(cfg.split_dirs["train"]) / f"{r}.h5" for r in run_ids]
        factors = build_norm_factors(train_df, cfg.stat_calc_columns, cfg.norm_by_col)
        raw = calc_scaler_stats(files, run_ids, cfg.stat_calc_columns, cfg.ts_agg, resolution=cfg.resolution, norm_factors=factors)
        stats_dir.mkdir(parents=True, exist_ok=True)
        path = make_unique_stat_file_name(cfg.resolution, cfg.norm_method, cfg.norm_by_col, cfg.stat_calc_columns, run_ids,stats_dir)
        save_stats(path, cfg.resolution, cfg.stat_calc_columns, raw, norm_by_col=cfg.norm_by_col)
        return {col: {k: float(v[i]) for k, v in raw.items()} for i, col in enumerate(cfg.stat_calc_columns)}

    def save_files(self, path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.cfg.save(path / "data_config.json")
        (path / "stats.json").write_text(json.dumps(self.stats, indent=2))
        joblib.dump(self.preprocessor, path / "preprocessor.joblib")
        (path / "meta.json").write_text(json.dumps({
            "split_hash": self.split_hash, "missing": self.missing,
            "n_runs": {k: len(v) for k, v in self.split_dfs.items()}}, indent=2))
        print(f"saved data pipeline (split {self.split_hash}) to {path}")

    @classmethod
    def load_files(cls, path, manifest, split_dirs=None, stats_dir=None):
        path = Path(path)
        cfg = DataConfig.load(path / "data_config.json")
        if split_dirs:
            cfg.split_dirs = {**cfg.split_dirs, **{k: str(v) for k, v in split_dirs.items()}}
        if stats_dir:
            cfg.stats_dir = str(stats_dir)
        meta = json.loads((path / "meta.json").read_text())
        full = make_splits(manifest, cfg.strat_cols, cfg.split_params)

        if hash_checker(full) != meta["split_hash"]:
            raise ValueError(f"recomputed split {hash_checker(full)} != stored {meta['split_hash']}")
        filtered = {name: existing_filter(df, cfg.split_dirs[name])[0] for name, df in full.items()}
        stats = json.loads((path / "stats.json").read_text())
        return cls(cfg, filtered, joblib.load(path / "preprocessor.joblib"), stats, meta["split_hash"], meta["missing"])

    @property
    def cat_cardinalities(self):
        return [len(cats) for cats in self.preprocessor.cat_encoder.categories_]

    def model_kwargs(self):
        return dict(
                    channels=len(self.cfg.target_cols),
                    cat_cardinalities=self.cat_cardinalities,
                    n_cont=len(self.cfg.static_cont_cols),
                    n_bool=len(self.cfg.static_bool_cols)+ self.cfg.n_static_flags,
                    n_dyn_cond_cols=self.cfg.n_dyn_model_cols)

    def x0_minmax(self):
        return scaled_minmax(self.stats, self.cfg.target_cols, self.target_scaler)


    # Dataloader and Sets

    def prepare_data(self, split):
        cache = self.__dict__.setdefault("prepared_data", {})
        if split not in cache:
            df = self.split_dfs[split]
            files, run_ids, static = normalizestatic(df, self.cfg.split_dirs[split], self.preprocessor)
            cache[split] = dict(files=files, run_ids=run_ids, static=static, target_norm=build_norm_factors(df, self.cfg.target_cols, self.cfg.norm_by_col),
                                condition_norm=build_norm_factors(df, self.cfg.dynamic_cond_cols, self.cfg.norm_by_col))
        return cache[split]

    def fill_dataset(self, split, augment):
        cfg = self.cfg
        prep = self.prepare_data(split)
        return dict(
            gen_cols= cfg.target_cols,
            cond_cols= cfg.dynamic_cond_cols,
            window_length= cfg.window_length,
            stride = cfg.stride,
            target_scaler = self.target_scaler,
            condition_scaler = self.condition_scaler,
            static_by_run_id = prep["static"],
            resolution=cfg.resolution,
            aggregation=cfg.ts_agg,
            target_norm=prep["target_norm"],
            condition_norm=prep["condition_norm"],
            context_steps=cfg.context_steps,
            context_reverse=cfg.context_reverse,
            use_season=cfg.use_season,
            context_dropout=cfg.context_dropout if augment else 0.0,
            context_noise=cfg.context_noise if augment else 0.0
        )

    def dataset(self, split, train=False):
        prep = self.prepare_data(split)
        return h5Dataset(
            files=prep["files"],
            rund_ids=prep["run_ids"],
            shufflefiles=train,
            shufflewindows=train,
            buffer_size=self.cfg.buffer_size if train else 0,
            **self.fill_dataset(split, augment=train)
        )


    def Window_drawer(self, split, n_runs, n_windows, seed=0):
        return draw_sampling_windows(self.split_dfs[split], self.cfg.strat_cols, n_runs, n_windows,
                                     self.cfg.steps_per_day, seed=seed, window_days=self.cfg.window_days)

    def windowdataset(self, split, index):
        prep = self.prepare_data(split)
        return SampleWindowDataset(index=index,split_files=prep["files"], **self.fill_dataset(split, augment=False))

    def dataloader(self, dataset, batchsize=32, train=False, num_workers=4):
        param_dict = dict(batch_size=batchsize, num_workers=num_workers, drop_last=train, pin_memory=True, timeout=0)
        if num_workers >0:
            param_dict.update(persistent_workers=True,prefetch_factor=2, multiprocessing_context="spawn")
        return DataLoader(dataset, **param_dict)

    def static_condition(self, params, batch_size, split="train", device="cpu"):
        return build_condition(self.preprocessor, self.split_dfs[split], params, batch_size, device)

def split_condition(self, cond):
    n = len(self.cfg.dynamic_cond_cols)
    season = 2 if self.cfg.use_season else 0
    return dict(dynamic=cond[:, :n], season=cond[:, n:n + season] if season else None,
                context=cond[:, n + season:-1] if self.cfg.context_steps else None,
                mask=cond[:, -1] if self.cfg.context_steps else None)