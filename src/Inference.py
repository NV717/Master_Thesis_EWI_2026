import json
import math
import zlib
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from src.Diffusion1d import sample1d, sample_year
from src.DataSets import apply_scaler, invert_scaler, scale_dynamic, split_windows
from tqdm.auto import tqdm


def original_units(x, scaler, norm):
    return (invert_scaler(x, scaler) *norm).cpu().numpy()




@torch.inference_mode()
def generate_windows(pipe, model, schedule, cond, split, index, n_samples=1, batch_size=32,
                     sampler="dpm", num_steps=50, seed=0, clip_range="auto",keep=None, device="cuda"):
    cfg = pipe.cfg
    length = cfg.window_length
    n_trgt_cols = len(cfg.target_cols)
    n_dyn = len(cfg.dynamic_cond_cols)
    s0 = n_dyn + (2 if cfg.use_season else 0)
    if isinstance(clip_range, str) and clip_range == "auto":
        clip_range = pipe.x0_minmax()

    dl = pipe.dataloader(pipe.windowdataset(split, index), batchsize=batch_size, num_workers=0)
    ts_step = pd.Timedelta(cfg.resolution).value
    frames = []

    total = -(-len(index) // batch_size)
    for i, batch in enumerate(tqdm(dl,total=total)):
        dyn = batch["condition"].to(device)
        static = {k: batch[k].to(device) for k in ("static_cat", "static_cont", "static_bool")}
        B = dyn.shape[0]
        gen = []

        for s in range(n_samples):
            torch.manual_seed(seed + 10000 * i + s)
            x = sample1d(model, length, B, n_trgt_cols, schedule, **static, dyn=dyn, cond=cond,
                         clip_range=clip_range, sampler=sampler, num_steps=num_steps, keep=keep)

            gen.append(original_units(x, pipe.target_scaler, batch["target_norm"]))

        real = original_units(batch["target"], pipe.target_scaler, batch["target_norm"])
        conds = original_units(batch["condition"][:, :n_dyn], pipe.condition_scaler, batch["condition_norm"])
        has_ctx = (batch["condition"][:, s0 + n_trgt_cols, 0] > 0).numpy() if cfg.context_steps else np.zeros(B, bool)

        meta = index.loc[batch["row"].numpy()]
        start = np.repeat(meta["start_hour"].to_numpy(), length)
        step = np.tile(np.arange(length), B)
        base = {
                    "run_id": np.repeat(meta["run_id"].to_numpy(), length),
                    "window_start": start,
                    "step": step.astype(np.int32),
                    "ts": pd.to_datetime(np.repeat(meta["year"].astype(str).to_numpy(), length) + "-01-01")
                          + pd.to_timedelta((start + step) * ts_step, unit="ns"),
                    "has_ctx": np.repeat(has_ctx, length),
                    **{c: conds[:, k].reshape(-1) for k, c in enumerate(cfg.dynamic_cond_cols)},
                }

        K = cfg.context_steps
        ctx = batch["condition"][:, s0:s0 + n_trgt_cols, :K]
        ctx = original_units(ctx, pipe.target_scaler, batch["target_norm"])
        if cfg.context_reverse:
            ctx = ctx[..., ::-1]
        full = np.full((B, n_trgt_cols, length), np.nan, dtype=np.float32)
        full[:, :, :K] = ctx
        full[~has_ctx] = np.nan

        base.update({f"ctx_{c}": full[:, k].reshape(-1) for k, c in enumerate(cfg.target_cols)})

        for sid, arr in [(-1, real), *enumerate(gen)]:
            frames.append(pd.DataFrame({**base, "sample_id": np.int16(sid),
                                        **{c: arr[:, k].reshape(-1) for k, c in enumerate(cfg.target_cols)}}))

    return pd.concat(frames, ignore_index=True)


def year_conditions(ds, run_id, path, window_length):
    target, condition_raw, season = ds._read_series(path)
    cond = scale_dynamic(condition_raw, ds.condition_norm[run_id], ds.condition_scaler, season)
    days, run_length = split_windows(cond, window_length)
    return target, condition_raw, days, run_length

# todo batch several runs together
@torch.inference_mode()
def generate_years(pipe, model, schedule, cond, split, run_ids,n_samples=8, sampler="dpm",
                   num_steps=100, seed=0, clip_range="auto", keep=None, device="cuda", outdir=None):
    cfg = pipe.cfg
    window_length = cfg.window_length
    n_target_cols = len(cfg.target_cols)
    if isinstance(clip_range, str) and clip_range == "auto":
        clip_range = pipe.x0_minmax()

    ds = pipe.dataset(split)
    prep = pipe.prepare_data(split)
    path_by_run = dict(zip(prep["run_ids"], prep["files"]))
    year_by_run = pipe.split_dfs[split].set_index("run_id")["year"]
    frames = []

    for run_id in tqdm(run_ids):
        out = Path(outdir) / f"run_id={run_id}" / "part.parquet" if outdir else None
        if out is not None and out.exists():
            continue

        target, condition_raw, days, run_length = year_conditions(ds, run_id, path_by_run[run_id], window_length)
        dyn_days = [d.repeat(n_samples, 1, 1).to(device) for d in days]
        stat_vals = prep["static"][run_id]
        static_cat=stat_vals["categorical"].unsqueeze(dim=0).repeat(n_samples, 1).to(device)
        static_cont=stat_vals["continuous"].unsqueeze(dim=0).repeat(n_samples, 1).to(device)
        static_bool=stat_vals["boolean"].unsqueeze(dim=0).repeat(n_samples, 1).to(device)

        torch.manual_seed(seed + zlib.crc32(run_id.encode())) #todo use hashlib insted?

        y = sample_year(model,
                        schedule,
                        dyn_days,
                        static_cat=static_cat,
                        static_cont=static_cont,
                        static_bool=static_bool,
                        context_steps=cfg.context_steps,
                        n_tgt=n_target_cols,
                        length=window_length,
                        reverse=cfg.context_reverse,
                        clip_range=clip_range,
                        return_cond=False,
                        cond=cond,
                        sampler=sampler,
                        num_steps=num_steps,
                        keep=keep)

        gen = (invert_scaler(y, pipe.target_scaler) * torch.from_numpy(ds.target_norm[run_id]))[..., :run_length].numpy()

        ts = pd.date_range(f"{year_by_run[run_id]}-01-01", periods=run_length, freq=cfg.resolution)
        df = output_df(run_id, ts, window_length, condition_raw, [(-1, target), *enumerate(gen)], cfg)

        if out is None:
            frames.append(df)
        else:
            out.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(out, compression="zstd", index=False)

    return pd.concat(frames, ignore_index=True) if outdir is None else None

def output_df(run_id, ts, window_length, conditions, series, cfg, has_ctx=None):
    run_length = len(ts)
    steps = np.arange(run_length)
    windows = steps // window_length
    base = {
        "run_id": run_id, "win_idx": windows.astype(np.int16),
        "window_start": windows * window_length, "step": (steps % window_length).astype(np.int32),
        "ts": ts, "has_ctx": (windows > 0) if has_ctx is None else has_ctx,
        **{c: conditions[k] for k, c in enumerate(cfg.dynamic_cond_cols)},
    }

    return pd.concat([pd.DataFrame({**base, "sample_id": np.int16(sid),**{c: arr[k] for k, c in enumerate(cfg.target_cols)}}) for sid, arr in series], ignore_index=True)


def write_inference(df, root, *, meta):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    df.to_parquet(root, partition_cols=["run_id"], compression="zstd", index=False)
    (root / "meta.json").write_text(json.dumps(meta, indent=2, default=str))


@torch.inference_mode()
def custom_sampling(pipe, model, schedule, cond, params, dynamic, n_samples=8, drop=(), name="custom",
                    sampler="dpm", num_steps=50, seed=0,context=None ,clip_range="auto", strict=True, device="cuda"):
    cfg = pipe.cfg
    window_length = cfg.window_length
    n_trgt_cols = len(cfg.target_cols)
    if isinstance(clip_range, str) and clip_range == "auto":
        clip_range = pipe.x0_minmax()

    conditions = pipe.custom_condition(params, dynamic, n_samples=n_samples, strict=strict, drop=drop,context=context, device=device)
    dyn_days = [w.repeat(n_samples, 1, 1).to(device) for w in conditions["windows"]]

    torch.manual_seed(seed + zlib.crc32(name.encode()))

    year = sample_year(model,
                       schedule,
                       dyn_days,
                       conditions["static_cat"],
                       conditions["static_cont"],
                       conditions["static_bool"],
                       cfg.context_steps,
                       n_trgt_cols,
                       window_length,
                       reverse=cfg.context_reverse,
                       clip_range=clip_range,
                    return_cond=False,
                       cond=cond,
                       keep=conditions["keep"] or None,
                       init_ctx=conditions["init_ctx"],
                    sampler=sampler,
                       num_steps=num_steps)

    gen = (invert_scaler(year, pipe.target_scaler) * torch.from_numpy(conditions["target_norm"]))[..., :conditions["n_steps"]].numpy()

    shown = conditions["dynamic"].copy()
    dropped_cols = [col for g in drop if g in cfg.cond_groups and g != "season" for col in cfg.cond_groups[g]]
    if dropped_cols:
        shown[dropped_cols] = np.nan  # placeholders are not real values

    has_ctx = None if context is None else np.ones(conditions["n_steps"], bool)

    return output_df(name, conditions["index"], window_length, shown.to_numpy(np.float32).T, list(enumerate(gen)), cfg,has_ctx=has_ctx)
