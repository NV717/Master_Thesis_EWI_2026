import torch
from tqdm.auto import tqdm
import hashlib, json, os
from datetime import datetime
from pathlib import Path
from src.Diffusion1d import p_losses

def train(model, schedule, dataloader, optimizer, ema, device, *,
          epochs, loss_type="l2", run=None, log_every=100,
          run_dir=None, config=None, extras=None, save_every=1,
          start_epoch=0, global_step=0, val_raw=False,
          val_dataloader=None, train_eval_dataloader=None,
          val_every_steps=None, val_batches=None, ckpt_every_steps=None):
    schedule = schedule.to(device)
    epoch_losses = []

    def run_validation(epoch, global_step):
        val_loss, val_per_t = validate(ema.ema_model, schedule, val_dataloader, device,
                                       loss_type=loss_type, n_batches=val_batches)
        print(f"step {global_step}: val_loss_ema={val_loss:.4f}")
        if run is not None:
            run.log({"val/loss_ema": val_loss,
                     **{f"val/loss_ema_t{t}": v for t, v in val_per_t.items()},
                     "epoch": epoch + 1, "global_step": global_step})
        if train_eval_dataloader is not None:
            train_eval, train_eval_per_t = validate(ema.ema_model, schedule, train_eval_dataloader, device,
                                     loss_type=loss_type, n_batches=val_batches)
            if run is not None:
                run.log({"train_eval/loss_ema": train_eval,
                         **{f"train_eval/loss_ema_t{t}": v for t, v in train_eval_per_t.items()},
                         "epoch": epoch + 1, "global_step": global_step})

        if val_raw:
            raw_loss, _ = validate(model, schedule, val_dataloader, device,
                                   loss_type=loss_type, n_batches=val_batches)
            model.train()
            if run is not None:
                run.log({"val/loss_raw": raw_loss, "epoch": epoch + 1, "global_step": global_step})



    def checkpoint(path, epochs_done, global_step):
        save_checkpoint(path, model=model, ema=ema, optimizer=optimizer, epoch=epochs_done,
                        global_step=global_step, config=config, extras=extras)

    for epoch in range(start_epoch, epochs):
        model.train()
        running_loss, num_batches = 0.0, 0

        for step, batch in enumerate(tqdm(dataloader, desc=f"epoch {epoch + 1}/{epochs}")):
            x = batch["target"].to(device)
            dyn = batch["condition"].to(device)
            static_cat = batch["static_cat"].to(device)
            static_cont = batch["static_cont"].to(device)
            static_bool = batch["static_bool"].to(device)

            t = torch.randint(0, schedule.timesteps, (x.shape[0],), device=device).long()

            optimizer.zero_grad()
            loss = p_losses(model, schedule, x, t, static_cat, static_cont, static_bool,
                            dyn=dyn, loss_type=loss_type)
            loss.backward()
            optimizer.step()
            ema.update()

            l = loss.item()
            running_loss += l
            num_batches += 1

            if step % log_every == 0:
                print(f"Loss: {l:.4f}")
            if run is not None:
                run.log({
                    "train/loss": l,
                    "train/learning_rate": optimizer.param_groups[0]["lr"],
                    "epoch": epoch + 1,
                    "global_step": global_step,
                })
            global_step += 1

            if val_dataloader is not None and val_every_steps and global_step % val_every_steps == 0:
                run_validation(epoch, global_step)
            if run_dir is not None and ckpt_every_steps and global_step % ckpt_every_steps == 0:
                checkpoint(run_dir / "last.pt", epoch, global_step)   # epoch not finished yet

        epoch_loss = running_loss / num_batches
        epoch_losses.append(epoch_loss)
        print(f"Epoch {epoch + 1}: loss={epoch_loss:.4f}")
        if run is not None:
            run.log({"epoch/loss": epoch_loss, "epoch": epoch + 1})

        if run_dir is not None:
            checkpoint(run_dir / "last.pt", epoch + 1, global_step)
            if (epoch + 1) % save_every == 0:
                checkpoint(run_dir / f"epoch_{epoch + 1:03d}.pt", epoch + 1, global_step)

    if run_dir is not None:
        checkpoint(run_dir / "final.pt", epochs, global_step)
    return {"epoch_losses": epoch_losses, "global_step": global_step}

# def train(model, schedule, dataloader, optimizer, ema, device, *,
#           epochs, loss_type="l2", run=None, log_every=100,run_dir=None, config=None, extras=None, save_every=1,
#           start_epoch=0,  global_step=0,train_eval_dataloader=None, val_dataloader=None, val_every=1, val_batches=None):
#     schedule = schedule.to(device)
#     epoch_losses = []
#
#     for epoch in range(start_epoch,epochs):
#         model.train()
#         running_loss = 0.0
#         num_batches =  0
#
#         for step, batch in enumerate(tqdm(dataloader, desc=f"epoch {epoch + 1}/{epochs}")):
#             x = batch["target"].to(device)
#             dyn = batch["condition"].to(device)
#             static_cat = batch["static_cat"].to(device)
#             static_cont = batch["static_cont"].to(device)
#             static_bool = batch["static_bool"].to(device)
#
#             t = torch.randint(0, schedule.timesteps, (x.shape[0],), device=device).long()
#
#             optimizer.zero_grad()
#             loss = p_losses(model, schedule, x, t, static_cat, static_cont, static_bool,
#                             dyn=dyn, loss_type=loss_type)
#             loss.backward()
#             optimizer.step()
#             ema.update()
#
#             l = loss.item()
#             running_loss += l
#             num_batches += 1
#
#             if step % log_every == 0:
#                 print(f"Loss: {l:.4f}")
#             if run is not None:
#                 run.log({
#                     "train/loss": l,
#                     "train/learning_rate": optimizer.param_groups[0]["lr"],
#                     "epoch": epoch + 1,
#                     "global_step": global_step,
#                 })
#             global_step += 1
#
#         epoch_loss = running_loss / num_batches
#         epoch_losses.append(epoch_loss)
#         print(f"Epoch {epoch + 1}: loss={epoch_loss:.4f}")
#         if run is not None:
#             run.log({"epoch/loss": epoch_loss, "epoch": epoch + 1})
#         # save checkpoint
#         if run_dir is not None:
#             kw = dict(model=model, ema=ema, optimizer=optimizer, epoch=epoch + 1,
#                       global_step=global_step, config=config, extras=extras)
#             save_checkpoint(run_dir / "last.pt", **kw)
#             if (epoch + 1) % save_every == 0:
#                 save_checkpoint(run_dir / f"epoch_{epoch + 1:03d}.pt", **kw)
#         # validation losses on fixed dataset
#         def run_validation():
#             val_loss, val_per_t = validate(ema.ema_model, schedule, val_dataloader, device,
#                                            loss_type=loss_type, n_batches=val_batches)
#             if run is not None:
#                 run.log({"val/loss_ema": val_loss,
#                          **{f"val/loss_ema_t{t}": v for t, v in val_per_t.items()},
#                          "epoch": epoch + 1, "global_step": global_step})
#             return val_loss
#
#         # inside the batch loop, after global_step += 1
#         if val_dataloader is not None and val_every and global_step % val_every == 0:
#             run_validation()
#             model.train()  # needed here: the raw model is validated, or any other eval() switch
#
#
#         if train_eval_dataloader is not None:
#             train_eval, _ = validate(ema.ema_model, schedule, train_eval_dataloader, device,
#                                      loss_type=loss_type, n_batches=val_batches)
#             if run is not None:
#                 run.log({"train_eval/loss_ema": train_eval, "epoch": epoch + 1, "global_step": global_step})
#
#
#     if run_dir is not None:
#         save_checkpoint(run_dir / "final.pt", model=model, ema=ema, optimizer=optimizer,
#                         epoch=epochs, global_step=global_step, config=config, extras=extras)
#     return {"epoch_losses": epoch_losses, "global_step": global_step}

def make_run_dir(config, root="outputs/runs"):
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    h = hashlib.sha256(json.dumps(config, sort_keys=True, default=str).encode()).hexdigest()[:6]
    run_dir = Path(root) / f"{stamp}_{h}"
    run_dir.mkdir(parents=True)
    (run_dir / "config.json").write_text(json.dumps(config, indent=2, default=str))
    return run_dir

def save_checkpoint(path, *, model, ema, optimizer, epoch, global_step, config, extras=None):
    path = Path(path)
    tmp = path.with_suffix(".tmp")
    torch.save({
        "model": model.state_dict(),
        "ema": ema.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "config": config,
        "extras": extras or {},
    }, tmp)
    os.replace(tmp, path)

def load_checkpoint(path, model=None, ema=None, optimizer=None, map_location="cpu"):
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    if model is not None:     model.load_state_dict(ckpt["model"])
    if ema is not None:       ema.load_state_dict(ckpt["ema"])
    if optimizer is not None: optimizer.load_state_dict(ckpt["optimizer"])
    return ckpt


@torch.inference_mode()
def validate(model, schedule, dataloader, device, *, loss_type="l2",
             n_batches=None, n_t=10, seed=0):
    model.eval()
    gen = torch.Generator(device=device).manual_seed(seed)
    t_grid = torch.linspace(0, schedule.timesteps - 1, n_t).long().tolist()
    sums, count = {t: 0.0 for t in t_grid}, 0

    for i, batch in enumerate(dataloader):
        if n_batches is not None and i >= n_batches:
            break
        x = batch["target"].to(device)
        dyn = batch["condition"].to(device)
        sc, sn, sb = (batch[k].to(device) for k in ("static_cat", "static_cont", "static_bool"))

        for t_val in t_grid:
            t = torch.full((x.shape[0],), t_val, device=device, dtype=torch.long)
            noise = torch.randn(x.shape, device=device, generator=gen)
            sums[t_val] += p_losses(model, schedule, x, t, sc, sn, sb, noise=noise,
                                    dyn=dyn, loss_type=loss_type).item()
        count += 1

    per_t = {t: s / count for t, s in sums.items()}
    return sum(per_t.values()) / len(per_t), per_t


