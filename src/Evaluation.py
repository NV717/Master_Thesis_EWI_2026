import pandas as pd
import numpy as np
from joblib import Parallel, delayed
from src.DataPreparation import _yearly_stats, _daily_stats, _hourly_profile, LOAD_DOMAINS, OCC_COL, OFFDAY_COL
from src.DataSets import build_norm_factors

def decode_static(pp, cat, cont, boolean):
    cat_df = pd.DataFrame(pp.cat_encoder.inverse_transform((cat.cpu().numpy() - 1)),
                          columns=pp.static_cat_cols)
    cont_df = pd.DataFrame(pp.cont_scaler.inverse_transform(cont.cpu().numpy()),
                           columns=pp.static_cont_cols)
    bool_df = pd.DataFrame(boolean.cpu().numpy() > 0.5, columns=pp.static_bool_cols)
    return pd.concat([cat_df, cont_df, bool_df], axis=1)


def series_stats(ts, total_persons, a_ref, freq, yearly):
    off_steps = ts[OFFDAY_COL].astype(bool)
    off_days = off_steps.groupby(ts.index.date).max()
    out = {"yearly": [], "daily": [], "hourly": []}
    for domain, col in LOAD_DOMAINS.items():
        load = ts[col]
        if yearly:
            out["yearly"].append({"domain": domain, **_yearly_stats(load, ts[OCC_COL], total_persons, a_ref, freq)})
        out["daily"].extend({"domain": domain, **r} for r in _daily_stats(load, off_days, freq))
        out["hourly"].extend({"domain": domain, **r} for r in _hourly_profile(load, off_steps))
    return out


def compute_metrics(df, manifest, freq, mode, n_jobs=-1):
    info = manifest.set_index("run_id")[["n_persons_sum", "a_ref"]]
    keys = ["run_id", "sample_id"] + (["window_start"] if mode == "window" else [])

    def one(key, g):
        ts = g.sort_values("ts").set_index("ts")
        rid = key[0]
        res = series_stats(ts, info.at[rid, "n_persons_sum"], info.at[rid, "a_ref"], freq, yearly=(mode == "year"))
        tag = dict(zip(keys, key))
        return {k: [{**tag, **row} for row in rows] for k, rows in res.items()}

    parts = Parallel(n_jobs=n_jobs)(delayed(one)(k, g) for k, g in df.groupby(keys))
    return {k: pd.DataFrame([row for p in parts for row in p[k]]) for k in ("yearly", "daily", "hourly")}



def clip_bounds(pipe, manifest):
    cfg = pipe.cfg
    norms = build_norm_factors(manifest, cfg.target_cols, cfg.norm_by_col)
    hi = np.array([pipe.stats[c]["max"] for c in cfg.target_cols])
    return pd.DataFrame({rid: n.ravel() * hi for rid, n in norms.items()}, index=cfg.target_cols).T


def check_timeseries_generated(df, target_cols, clip_hi, window_length, edge_steps,
                               keys=("run_id", "sample_id", "window_start"),off_threshold=None):
    keys = [k for k in keys if k in df.columns]
    out = []
    for col in target_cols:
        x = df[col]
        hi = df["run_id"].map(clip_hi[col])
        extrem_peak = x >= 0.999 * hi
        near = (x <= df["run_id"].map(off_threshold[col])) if off_threshold is not None and col in off_threshold else pd.Series(False,index=df.index)
        edge = (df["step"] < edge_steps) | (df["step"] >= window_length - edge_steps)
        flags = pd.DataFrame({
            **{k: df[k] for k in keys},
            "x": x, "nan": x.isna(), "inf": np.isinf(x), "neg": x  < -1e-6, "zero": x == 0,
            "extrem_peak": extrem_peak, "extrem_peak_edge": extrem_peak & edge, "hi": hi,"near": near
        })
        g = flags.groupby(keys, sort=False)
        res = g.agg(n=("x", "size"), n_nan=("nan", "sum"), n_inf=("inf", "sum"), n_neg=("neg", "sum"),
                    min=("x", "min"), max=("x", "max"), mean=("x", "mean"), std=("x", "std"),
                    zero_share=("zero", "mean"), extrem_peak_share=("extrem_peak", "mean"),
                    n_extrem_peak=("extrem_peak", "sum"), n_extrem_peak_edge=("extrem_peak_edge", "sum"), hi=("hi", "first"), near_zero_share=("near", "mean"),)
        res["max_over_clip"] = res["max"] / res["hi"]
        res["extrem_peak_edge_share"] = (res["n_extrem_peak_edge"] / res["n_extrem_peak"]).where(res["n_extrem_peak"] > 0)
        res["const"] = res["std"].fillna(0) == 0
        res.insert(0, "column", col)
        out.append(res.drop(columns=["hi", "n_extrem_peak", "n_extrem_peak_edge"]).reset_index())
    return pd.concat(out, ignore_index=True)



def check_timeseries_summary(s):
    s = s.assign(kind=np.where(s["sample_id"] == -1, "real", "gen"))
    return s.groupby(["column", "kind"]).agg(
        series=("n", "size"),
        with_nan=("n_nan", lambda v: int((v > 0).sum())),
        with_inf=("n_inf", lambda v: int((v > 0).sum())),
        with_neg=("n_neg", lambda v: int((v > 0).sum())),
        const=("const", "sum"),
        min=("min", "min"), max=("max", "max"),
        max_over_clip_p95=("max_over_clip", lambda v: v.quantile(0.95)),
        extrem_peak_share_mean=("extrem_peak_share", "mean"),
        extrem_peak_edge_share_mean=("extrem_peak_edge_share", "mean"),
        zero_share_mean=("zero_share", "mean"),
        near_zero_share_mean=("near_zero_share", "mean")
    )

OFF_RULES = {"Hot Water Load": ("abs", 0.1), "Heating Load": ("per_a_ref", 5e-4)}

def off_thresholds(manifest, run_ids, rules=OFF_RULES):
    m = manifest.drop_duplicates("run_id").set_index("run_id").loc[list(pd.unique(run_ids))]
    out = pd.DataFrame(index=m.index)
    for col, (kind, v) in rules.items():
        out[col] = v if kind == "abs" else v * m["a_ref"]
    return out

def snap_to_zero(df, off_thr, cols=("Hot Water Load",)):
    out = df.copy()
    for col in cols:
        x = out[col].clip(lower=0)
        out[col] = x.where(x >= out["run_id"].map(off_thr[col]), 0.0)


    return out


