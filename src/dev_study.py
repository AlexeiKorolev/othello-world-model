"""Developmental study: train the normal (GELU) Othello-GPT while tracking,
every other epoch, when the Nanda linear "mine/yours" world model arises.

At epoch 0 (random init) and every `dev_study.probe_every_n_epochs` epochs
thereafter, this pauses training to: extract mine/yours-labelled activations
on a held-out sample (nanda_probe.extract_myours), train small linear probes
(train_probe.train_one_probe), and run the causal mine/yours intervention
benchmark (same loop as intervene_nanda.py) using those probes. Settings are
deliberately lighter than the end-of-training scripts (see configs/config.yaml
dev_study section) since this runs ~20x in one job instead of once.

After training, the FULL-scale nanda_probe.py and intervene_nanda.py run once
on the final checkpoint unchanged, so the usual polished artifacts
(nanda_probe_<profile>.json, intervention_nanda_<profile>.json) land exactly
as if `sbatch slurm/nanda_probe.slurm` / `intervene_nanda.slurm` had been run
directly. This script's checkpoint IS the normal gpt_synthetic_<profile>.ckpt
-- this run doubles as "train GPT 1", not a side experiment.

Writes results/dev_study_<profile>.json (one record per probed epoch) and
results/dev_study_<profile>.png (probe error + intervention effect vs epoch).

    python dev_study.py --config ../configs/config.yaml --profile pilot
"""
import os
import sys
import json
import time
import pickle
import argparse
import subprocess

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import common
from train_gpt import evaluate_legal_move_error
from resumable_trainer import ResumableTrainer
from nanda_probe import extract_myours
from train_probe import train_one_probe


def run_light_probe_and_intervene(raw_model, mconf, val_games, bench, ds, cfg, device):
    """One extraction pass -> per-layer mine/yours linear probe error, then a
    causal intervention pass reusing those same probes. Mirrors nanda_probe.py
    + intervene_nanda.py, at whatever scale `ds` specifies -- either the
    reduced cfg['dev_study'] settings (cheap, ~20x per training run) or the
    full cfg['probe']/cfg['intervene'] settings via --full_fidelity_probe
    (expensive, but directly comparable to nanda_probe.py/intervene_nanda.py's
    own numbers)."""
    from mingpt.model import GPTforProbeIA
    from mingpt.probe_model import BatteryProbeClassification
    from mingpt.utils import intervene as intervene_step
    from data.othello import OthelloBoardState, permit_reverse

    layers = ds["probe_layers"]
    layer_s, layer_e = ds["intervene_layer_s"], ds["intervene_layer_e"]
    n_embd = cfg["model"]["n_embd"]

    raw_model.eval()
    A, _Yabs, Y_my = extract_myours(raw_model, val_games, ds["probe_max_games"],
                                    layers, device)
    probes, errs = {}, {}
    for L in layers:
        p = BatteryProbeClassification(device, probe_class=3, num_task=64,
                                       input_dim=n_embd)
        errs[L] = train_one_probe(A[L], Y_my, p, ds["probe_epochs"],
                                  cfg["probe"]["batch_size"], device)
        probes[L] = p
    del A, Y_my

    model_ia = GPTforProbeIA(mconf, probe_layer=layer_s)
    model_ia.load_state_dict(raw_model.state_dict())
    model_ia = model_ia.to(device).eval()

    ic = cfg["intervene"]
    htd = {"lr": ic["lr"], "steps": ds["intervene_steps"], "reg_strg": ic["reg_strength"]}
    n_cases = min(ds["intervene_n_cases"], len(bench))

    def to_mine_yours(c_abs, side_black_to_move):
        return c_abs if side_black_to_move else (2 - c_abs)

    def legality_error(logits_vocab, true_legal):
        scores = [(common.ITOS[i], logits_vocab[i].item())
                  for i in range(1, common.VOCAB_SIZE)]
        scores.sort(key=lambda t: t[1], reverse=True)
        N = max(1, len(true_legal))
        pred = [m for m, _ in scores[:N]]
        return sum(1 for m in pred if m not in true_legal)

    intv_errs, base_errs = [], []
    for case in bench[:n_cases]:
        history = case["history"]; pos_int = case["pos_int"]; ori_color = case["ori_color"]
        intervention_to = 2 - ori_color
        side_black = (len(history) % 2 == 0)
        from_my = to_mine_yours(ori_color, side_black)
        to_my = to_mine_yours(intervention_to, side_black)
        wtd = {"intervention_position": permit_reverse(pos_int),
               "intervention_from": from_my, "intervention_to": to_my}

        g = torch.tensor([common.STOI[s] for s in history], dtype=torch.long, device=device)

        ab = OthelloBoardState(); ab.update(history)
        r, c = pos_int // 8, pos_int % 8
        ab.state[r, c] = intervention_to - 1
        true_legal = set(ab.get_valid_moves())

        with torch.no_grad():
            x0 = model_ia.forward_1st_stage(g[None, :])
            outs = model_ia.forward_2nd_stage(x0, layer_s, end_layer=-1)
            base_logits, _ = model_ia.predict(outs[-1])
        base_errs.append(legality_error(base_logits[0, -1], true_legal))

        whole = model_ia.forward_1st_stage(g[None, :])
        p = probes[layer_s]
        mid = whole[0, -1]
        labels = p(mid[None, :])[0].squeeze(0).argmax(dim=-1)
        new_mid = intervene_step(p, mid.clone(), labels, wtd, htd, plot=False)
        whole = whole.clone(); whole[0, -1] = new_mid
        for L in range(layer_s, layer_e - 1):
            whole = model_ia.forward_2nd_stage(whole, L, L + 1)[0]
            p = probes[L + 1]
            mid = whole[0, -1]
            labels = p(mid[None, :])[0].squeeze(0).argmax(dim=-1)
            new_mid = intervene_step(p, mid.clone(), labels, wtd, htd, plot=False)
            whole = whole.clone(); whole[0, -1] = new_mid
        with torch.no_grad():
            intv_logits, _ = model_ia.predict(whole)
        intv_errs.append(legality_error(intv_logits[0, -1], true_legal))

    raw_model.train()
    return errs, {
        "baseline_error_mean": float(np.mean(base_errs)),
        "intervened_error_mean": float(np.mean(intv_errs)),
    }


class DevStudyTrainer(ResumableTrainer):
    """ResumableTrainer + a hook fired at epoch 0 (random init) and after
    every completed epoch (the hook itself decides whether to act)."""
    def __init__(self, model, train_dataset, config, on_epoch):
        super().__init__(model, train_dataset, config)
        self.on_epoch = on_epoch

    def train(self, wall_budget_sec=None):
        raw = self._raw()
        optimizer = raw.configure_optimizers(self.config)
        self.maybe_resume(optimizer)

        if self.start_epoch == 0:
            self.on_epoch(0, raw)

        t0 = time.time(); ran = 0; epoch = self.start_epoch
        while epoch < self.config.max_epochs:
            self._run_epoch(optimizer, epoch)
            epoch += 1; ran += 1
            self.save_state(optimizer, epoch)
            self.on_epoch(epoch, raw)
            if epoch >= self.config.max_epochs:
                return True
            if wall_budget_sec is not None:
                elapsed = time.time() - t0
                per_epoch = elapsed / ran
                if elapsed + 1.2 * per_epoch > wall_budget_sec:
                    print(f"[dev_study] wall budget reached; pausing after epoch {epoch}",
                          flush=True)
                    return False
        return True


def plot_dev_study(results_path, res_dir, profile):
    with open(results_path) as f:
        study = json.load(f)
    recs = sorted(study["epochs"], key=lambda r: r["epoch"])
    epochs = [r["epoch"] for r in recs]
    layers = study["probe_layers"]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
    for L in layers:
        ys = [r["probe_myours_err"][str(L)] * 100 for r in recs]
        ax1.plot(epochs, ys, "o-", label=f"layer {L}")
    ax1.set_ylabel("mine/yours linear probe error (%)")
    ax1.set_title(f"When does the linear world model arise? ({profile}, GELU)")
    ax1.legend(); ax1.grid(alpha=0.3)

    base = [r["intervention_baseline_err"] for r in recs]
    intv = [r["intervention_intervened_err"] for r in recs]
    ax2.plot(epochs, base, "o-", color="#c44", label="baseline (no intervention)")
    ax2.plot(epochs, intv, "o-", color="#268", label="intervened")
    ax2.set_xlabel("epoch"); ax2.set_ylabel("legality error (top-N illegal count)")
    ax2.legend(); ax2.grid(alpha=0.3)

    plt.tight_layout()
    out = os.path.join(res_dir, f"dev_study_{profile}.png")
    plt.savefig(out, dpi=150)
    print(f"[dev_study] wrote {out}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--profile", default=None)
    ap.add_argument("--tag", default=None,
                    help="label for checkpoint/results filenames (default: profile); "
                         "use a distinct tag to run a second concurrent study on the "
                         "same profile/data without clobbering another run's outputs")
    ap.add_argument("--full_fidelity_probe", action="store_true",
                    help="use cfg['probe']/cfg['intervene'] settings (all layers in the "
                         "intervene range, full probe_epochs/max_probe_games/n_cases/steps) "
                         "for the interleaved per-epoch checks instead of the reduced "
                         "cfg['dev_study'] settings -- much more expensive per round, but "
                         "directly comparable to nanda_probe.py/intervene_nanda.py's numbers")
    args = ap.parse_args()

    cfg = common.load_config(args.config)
    if args.profile:
        cfg["profile"] = args.profile
    common.add_upstream_to_path(cfg)
    profile = cfg["profile"]
    tag = args.tag or profile
    ds_cfg = cfg["dev_study"]
    if args.full_fidelity_probe:
        ic = cfg["intervene"]
        ds_cfg = {
            "probe_every_n_epochs": ds_cfg["probe_every_n_epochs"],
            "probe_layers": list(range(ic["layer_s"], ic["layer_e"])),
            "probe_max_games": cfg["probe"]["max_probe_games"],
            "probe_epochs": cfg["probe"]["epochs"],
            "intervene_layer_s": ic["layer_s"],
            "intervene_layer_e": ic["layer_e"],
            "intervene_n_cases": ic["n_cases"],
            "intervene_steps": ic["steps"],
        }
    every = ds_cfg["probe_every_n_epochs"]
    cfg_path_used = args.config or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), os.pardir, "configs", "config.yaml")

    from mingpt.utils import set_seed
    from mingpt.model import GPT, GPTConfig
    from mingpt.trainer import TrainerConfig
    set_seed(cfg["train"]["seed"])

    device = "cuda" if torch.cuda.is_available() else "cpu"

    games = common.load_all_games(cfg["paths"]["data_dir"], profile)
    train_games, val_games = common.split_train_val(games, cfg["data"]["val_games"])
    if not os.path.exists(common.val_cache_path(cfg, profile)):
        common.save_val_cache(cfg, profile, val_games)  # deterministic split; don't
    # overwrite a cache another concurrent run under the same profile may be reading.
    print(f"[dev_study] {len(games)} games -> {len(train_games)} train / "
          f"{len(val_games)} val", flush=True)
    train_dataset = common.make_dataset(train_games)

    with open(os.path.join(cfg["paths"]["upstream"], "intervention_benchmark.pkl"), "rb") as f:
        bench = pickle.load(f)

    mconf = GPTConfig(common.VOCAB_SIZE, common.BLOCK_SIZE,
                      n_layer=cfg["model"]["n_layer"], n_head=cfg["model"]["n_head"],
                      n_embd=cfg["model"]["n_embd"])
    model = GPT(mconf)

    ckpt_dir = cfg["paths"]["ckpt_dir"]
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, f"gpt_synthetic_{tag}.ckpt")  # same naming train_gpt.py uses (tag defaults to profile)
    epoch_ckpt_dir = os.path.join(ckpt_dir, f"dev_study_{tag}")
    os.makedirs(epoch_ckpt_dir, exist_ok=True)

    res_dir = cfg["paths"]["results_dir"]
    os.makedirs(res_dir, exist_ok=True)
    results_path = os.path.join(res_dir, f"dev_study_{tag}.json")
    if os.path.exists(results_path):
        with open(results_path) as f:
            study = json.load(f)
    else:
        study = {"profile": profile, "tag": tag, "arch": "gelu",
                 "probe_layers": ds_cfg["probe_layers"],
                 "intervene_layer_s": ds_cfg["intervene_layer_s"],
                 "intervene_layer_e": ds_cfg["intervene_layer_e"],
                 "probe_every_n_epochs": every, "epochs": []}

    def on_epoch(epoch, raw_model):
        if epoch != 0 and epoch % every != 0:
            return
        t0 = time.time()
        errs, interv = run_light_probe_and_intervene(raw_model, mconf, val_games,
                                                      bench, ds_cfg, cfg, device)
        ck = os.path.join(epoch_ckpt_dir, f"gpt_epoch{epoch:03d}.ckpt")
        torch.save(raw_model.state_dict(), ck)
        record = {"epoch": epoch, "probe_myours_err": errs,
                  "intervention_baseline_err": interv["baseline_error_mean"],
                  "intervention_intervened_err": interv["intervened_error_mean"],
                  "ckpt": ck, "wall_s": round(time.time() - t0, 1)}
        study["epochs"].append(record)
        with open(results_path, "w") as f:
            json.dump(study, f, indent=2)
        best_L = min(errs, key=errs.get)
        print(f"[dev_study] epoch {epoch}: best mine/yours probe err "
              f"{errs[best_L]*100:.2f}% (L{best_L}) | intervention "
              f"{interv['baseline_error_mean']:.3f} -> {interv['intervened_error_mean']:.3f} "
              f"({time.time()-t0:.0f}s)", flush=True)

    max_epochs = common.profile_get(cfg, "train", "max_epochs")
    batch_size = common.profile_get(cfg, "train", "batch_size")
    block = train_dataset.block_size
    tconf = TrainerConfig(
        max_epochs=max_epochs, batch_size=batch_size,
        learning_rate=cfg["train"]["learning_rate"], betas=tuple(cfg["train"]["betas"]),
        grad_norm_clip=cfg["train"]["grad_norm_clip"], weight_decay=cfg["train"]["weight_decay"],
        lr_decay=True, warmup_tokens=len(train_dataset) * block * 5,
        final_tokens=len(train_dataset) * block * max_epochs,
        num_workers=cfg["train"]["num_workers"], ckpt_path=ckpt_path,
    )
    print(f"[dev_study] profile={profile} tag={tag} epochs={max_epochs} batch_size={batch_size} "
          f"probe_every={every} probe_layers={ds_cfg['probe_layers']}", flush=True)

    trainer = DevStudyTrainer(model, train_dataset, tconf, on_epoch)
    t0 = time.time()
    trainer.train()
    print(f"[dev_study] training done in {time.time()-t0:.0f}s -> {ckpt_path}", flush=True)

    # --- final eval, identical to a normal train_gpt.py run ---
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    model = model.to(device)
    err, ok, total = evaluate_legal_move_error(
        model, train_dataset, val_games, cfg["eval"]["n_val_games"], device)
    print(f"[dev_study] LEGAL-MOVE ERROR = {err*100:.4f}%  ({ok}/{total} legal)", flush=True)
    with open(os.path.join(res_dir, f"gpt_{tag}_eval.json"), "w") as f:
        json.dump({"profile": profile, "tag": tag, "arch": "gelu", "expansion": None,
                   "legal_move_error": err, "legal": ok, "total": total,
                   "ckpt": ckpt_path}, f, indent=2)

    # --- final FULL-scale nanda probe + intervention on the finished ckpt ---
    here = os.path.dirname(os.path.abspath(__file__))
    for script in ("nanda_probe.py", "intervene_nanda.py"):
        cmd = [sys.executable, script, "--config", cfg_path_used, "--profile", profile,
               "--ckpt", ckpt_path, "--tag", tag]
        print(f"[dev_study] running final full-scale stage: {' '.join(cmd)}", flush=True)
        subprocess.run(cmd, check=True, cwd=here)

    plot_dev_study(results_path, res_dir, tag)


if __name__ == "__main__":
    main()
