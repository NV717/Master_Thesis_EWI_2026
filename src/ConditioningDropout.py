import torch

class CondDropout:
    def __init__(self, group_idx, probs_group, probs_all=0.0, probs_cat=0.0,
                 probs_cont=0.0, ctx_mask_idx=None, static_flag=False, names=None):
        self.group_idx = [list(g) for g in group_idx]
        self.probs_group = list(probs_group)
        self.probs_all = probs_all
        self.probs_cat = probs_cat
        self.probs_cont = probs_cont
        self.ctx_mask_idx = ctx_mask_idx
        self.static_flag = static_flag
        self.names = list(names) if names is not None else []

    @classmethod
    def from_config(cls, cfg):
        return cls(cfg.dyn_group_idx.values(), cfg.dyn_group_probs, cfg.cond_dropout_all,cfg.static_dropout.get("cat", 0.0),
        cfg.static_dropout.get("cont", 0.0),cfg.context_mask_idx, static_flag=bool(cfg.n_static_flags), names=list(cfg.dyn_group_idx) )

    def keep_all(self, static_cat, value=1.0):
        B= static_cat.shape[0]
        n_cat= static_cat.shape[1]
        device = static_cat.device
        return dict(dyn=torch.full((B, len(self.group_idx)), value, device=device),
                    cat=torch.full((B, n_cat), value, device=device),
                    cont=torch.full((B,), value, device=device))

    def draw_dropout(self, static_cat):
        keep = self.keep_all(static_cat)
        B = static_cat.shape[0]
        device = static_cat.device
        p_dyn = torch.tensor(self.probs_group, device=device, dtype=torch.float32)
        keep["dyn"] = (torch.rand_like(keep["dyn"]) >= p_dyn).float()
        keep["cat"] = (torch.rand_like(keep["cat"]) >= self.probs_cat).float()
        keep["cont"] = (torch.rand_like(keep["cont"]) >= self.probs_cont).float()
        drop_all = torch.rand(B, device=device) < self.probs_all
        for k in keep:
            keep[k][drop_all]=0.0
        return keep

    def apply(self, dyn, static_cat, static_cont, static_bool, keep):
        if dyn is not None and self.group_idx:
            out = dyn.clone()
            for g, idx in enumerate(self.group_idx):
                out[:, idx] = out[:, idx] * keep["dyn"][:, g, None, None]
            flags = keep["dyn"][:, :, None].expand(-1, -1, dyn.shape[-1])
            dyn = torch.cat([out, flags], dim=1)
        static_cat = static_cat*keep["cat"].long()
        static_cont = static_cont * keep["cont"][:, None]
        static_bool = static_bool * keep["cont"][:, None]
        if self.static_flag:
            static_bool = torch.cat([static_bool, keep["cont"][:, None]], dim=1)
        return dyn, static_cat, static_cont, static_bool

    def keep_from(self, spec, static_cat):
        unknown = set(spec) -set(self.names) - {"static_cat", "static_cont"}
        if unknown:
            raise ValueError(f"unknown condition keys {unknown}, allowed: {self.names + ['static_cat', 'static_cont']}")
        keep = self.keep_all(static_cat)
        for g, name in enumerate(self.names):
            keep["dyn"][:, g] = float(spec.get(name, True))
        keep["cat"][:] = float(spec.get("static_cat", True))
        keep["cont"][:] = float(spec.get("static_cont", True))
        return keep


