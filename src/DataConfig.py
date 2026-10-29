from __future__ import annotations
import json
from dataclasses import dataclass, field, asdict, fields
from pathlib import Path
from src.DataSets import stepsperday

@dataclass
class DataConfig:
    target_cols: list
    dynamic_cond_cols: list
    static_cat_cols: list
    static_cont_cols: list
    static_bool_cols: list # share extra or give it here with?!
    resolution: str
    window_days: int
    stride_days: int
    norm_method: str = "none"
    norm_variant: dict = field(default_factory=lambda: {"none": {}})
    ts_agg: dict = field(default_factory=dict)
    scaler_by_col: dict = field(default_factory=dict)
    zero_cols: list = field(default_factory=list)
    context_steps: int=0
    context_reverse: bool=False
    use_season: bool = False
    context_dropout: float = 0.0
    context_noise: float = 0.0
    buffer_size: int = 4096
    strat_cols: list = field(default_factory=lambda: ["buildingType", "buildingAgeBin"])
    split_params: dict = field(default_factory=dict)
    split_dirs: dict = field(default_factory=dict)
    stats_dir: str = "../data/stats/scalerstats"
    cond_groups: dict = field(default_factory=dict)
    cond_dropout: dict = field(default_factory=dict)
    cond_dropout_all: float = 0.0
    static_dropout: dict = field(default_factory=dict)

    @property
    def norm_by_col(self):
        return self.norm_variant[self.norm_method]
    @property
    def stat_calc_columns(self):
        return self.target_cols + [c for c in self.dynamic_cond_cols if c not in self.target_cols]
    @property
    def steps_per_day(self):
        return stepsperday(self.resolution)
    @property
    def window_length(self):
        return   self.window_days*self.steps_per_day
    @property
    def stride(self):
        return self.stride_days*self.steps_per_day
    @property
    def n_context(self):
        return len(self.target_cols) + 1 if self.context_steps > 0 else 0
    @property
    def n_dyn_cond_cols(self):
        return len(self.dynamic_cond_cols) + (2 if self.use_season else 0) + self.n_context

    def to_dict(self):
        return json.loads(json.dumps(asdict(self), default=str))

    def save(self, path):
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def from_dict(cls,d):
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k,v in d.items() if k in known})

    @classmethod
    def load(cls, path):
        return cls.from_dict(json.loads(Path(path).read_text()))

    def __post_init__(self):
        seen = []
        for name, cols in self.cond_groups.items():
            if cols == "season":
                if not self.use_season:
                    raise ValueError("seson is n group")
                continue

            problem_cols = [c for c in cols if c not in self.dynamic_cond_cols or c in seen]
            if problem_cols:
                raise ValueError("col in multiple groups")
            seen += cols

        unkown =(set(self.cond_dropout) - set(self.cond_groups)) | (set(self.static_dropout) - {"cat", "cont"})
        if unkown:
            raise ValueError("dropout values dont match")

        probs = [*self.cond_dropout.values(), *self.static_dropout.values(), self.cond_dropout_all]

    @property
    def dyn_group_idx(self):
        n = len(self.dynamic_cond_cols)
        return {name: [n, n + 1] if cols == "season" else [self.dynamic_cond_cols.index(c) for c in cols] for name, cols in self.cond_groups.items()}

    @property
    def dyn_group_probs(self):
        return [self.cond_dropout.get(name, 0.0) for name in self.cond_groups]

    @property
    def context_mask_idx(self):
        if self.context_steps <= 0:
            return None
        return len(self.dynamic_cond_cols) + (2 if self.use_season else 0) + len(self.target_cols)

    @property
    def n_dyn_flags(self):
        return len(self.cond_groups)

    @property
    def n_static_flags(self):
        return 1 if (self.static_cont_cols or self.static_bool_cols) and self.static_dropout else 0

    @property
    def n_dyn_model_cols(self):
        return self.n_dyn_cond_cols + self.n_dyn_flags





