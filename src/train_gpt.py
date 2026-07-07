"""Train Othello-GPT from scratch on the synthetic dataset.

Headless port of upstream `train_gpt_othello.ipynb`. Same architecture
(8L/8H/512d), optimiser, and LR schedule. Uses the upstream Trainer, which
wraps the model in nn.DataParallel automatically, so requesting multiple GPUs
speeds it up. After training, evaluates the legal-move error rate by
autoregressive rollout over held-out validation games (paper's metric).

    python train_gpt.py --config ../configs/config.yaml --profile pilot
"""
import os
import sys
import json
import time
import argparse

import torch
from tqdm import tqdm

import common


def evaluate_legal_move_error(model, dataset, val_games, n_val, device):
    """Paper metric: at every prefix of each val game, greedily predict the
    next move and check it is legal. error = 1 - pass_rate."""
    from mingpt.utils import sample
    from data.othello import OthelloBoardState
    model.eval()
    total, ok = 0, 0
    bar = tqdm(val_games[:n_val], desc="legal-move eval", mininterval=30)
    for game in bar:
        for L in range(1, len(game)):
            total += 1
            context = game[:L]
            x = torch.tensor([dataset.stoi[s] for s in context],
                             dtype=torch.long)[None, ...].to(device)
            y = sample(model, x, 1, temperature=1.0)[0]
            completion = [dataset.itos[int(i)] for i in y if i != -1]
            try:
                OthelloBoardState().update(completion, prt=False)
            except Exception:
                pass
            else:
                ok += 1
        if total:
            bar.set_description(f"pass {ok/total*100:.2f}% ({ok}/{total})")
    err = 1.0 - ok / max(1, total)
    return err, ok, total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--profile", default=None)
    ap.add_argument("--eval-only", action="store_true",
                    help="skip training, just load ckpt and eval")
    ap.add_argument("--ckpt", default=None,
                    help="override checkpoint path (e.g. authors' pretrained)")
    ap.add_argument("--arch", default="gelu", choices=["gelu", "bilinear"],
                    help="MLP type: standard GELU or bilinear (arXiv:2410.08417)")
    ap.add_argument("--expansion", type=float, default=4.0,
                    help="bilinear d_hidden/d_model; 8/3~2.667 is param/FLOP-matched")
    ap.add_argument("--resume", action="store_true",
                    help="checkpoint/resume trainer: split training across jobs")
    ap.add_argument("--wall_hours", type=float, default=None,
                    help="with --resume: pause (exit 64) once the next epoch would "
                         "exceed this wall-clock budget, for the next chunk to resume")
    args = ap.parse_args()

    cfg = common.load_config(args.config)
    if args.profile:
        cfg["profile"] = args.profile
    common.add_upstream_to_path(cfg)
    profile = cfg["profile"]

    from mingpt.utils import set_seed
    from mingpt.model import GPT, GPTConfig
    from mingpt.trainer import Trainer, TrainerConfig
    set_seed(cfg["train"]["seed"])

    # --- data ---
    games = common.load_all_games(cfg["paths"]["data_dir"], profile)
    train_games, val_games = common.split_train_val(games, cfg["data"]["val_games"])
    common.save_val_cache(cfg, profile, val_games)   # downstream jobs reuse this
    print(f"[train] {len(games)} games -> {len(train_games)} train / "
          f"{len(val_games)} val", flush=True)
    train_dataset = common.make_dataset(train_games)

    # --- model ---
    mconf = GPTConfig(common.VOCAB_SIZE, common.BLOCK_SIZE,
                      n_layer=cfg["model"]["n_layer"],
                      n_head=cfg["model"]["n_head"],
                      n_embd=cfg["model"]["n_embd"])
    if args.arch == "bilinear":
        from bilinear_model import make_bilinear_gpt
        model = make_bilinear_gpt(mconf, expansion=args.expansion)
        print(f"[train] BILINEAR MLPs (expansion={args.expansion:.4f}); "
              f"params={sum(p.numel() for p in model.parameters())/1e6:.1f}M",
              flush=True)
    else:
        model = GPT(mconf)

    # bilinear checkpoints/results get an "_bilinear" suffix so they never clash
    # with the GELU baseline (same data + val split, different architecture).
    suffix = "" if args.arch == "gelu" else f"_{args.arch}"

    ckpt_dir = cfg["paths"]["ckpt_dir"]
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = args.ckpt or os.path.join(ckpt_dir,
                                          f"gpt_synthetic_{profile}{suffix}.ckpt")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if not args.eval_only:
        max_epochs = common.profile_get(cfg, "train", "max_epochs")
        batch_size = common.profile_get(cfg, "train", "batch_size")
        ngpu = torch.cuda.device_count()
        print(f"[train] profile={profile} epochs={max_epochs} "
              f"batch_size={batch_size} gpus={ngpu}", flush=True)
        block = train_dataset.block_size
        tconf = TrainerConfig(
            max_epochs=max_epochs,
            batch_size=batch_size,
            learning_rate=cfg["train"]["learning_rate"],
            betas=tuple(cfg["train"]["betas"]),
            grad_norm_clip=cfg["train"]["grad_norm_clip"],
            weight_decay=cfg["train"]["weight_decay"],
            lr_decay=True,
            warmup_tokens=len(train_dataset) * block * 5,
            final_tokens=len(train_dataset) * block * max_epochs,
            num_workers=cfg["train"]["num_workers"],
            ckpt_path=ckpt_path,
        )
        t0 = time.time()
        if args.resume:
            from resumable_trainer import ResumableTrainer
            budget = args.wall_hours * 3600 if args.wall_hours else None
            trainer = ResumableTrainer(model, train_dataset, tconf)
            complete = trainer.train(wall_budget_sec=budget)
            if not complete:
                print(f"[train] chunk paused after {time.time()-t0:.0f}s; "
                      f"exit 64 to resubmit next chunk", flush=True)
                sys.exit(64)
            print(f"[train] all epochs done in this/earlier chunks -> {ckpt_path}",
                  flush=True)
        else:
            trainer = Trainer(model, train_dataset, None, tconf)
            trainer.train()
            print(f"[train] training done in {time.time()-t0:.0f}s -> {ckpt_path}",
                  flush=True)

    # --- load best checkpoint + evaluate ---
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    model = model.to(device)
    err, ok, total = evaluate_legal_move_error(
        model, train_dataset, val_games, cfg["eval"]["n_val_games"], device)
    print(f"[train] LEGAL-MOVE ERROR = {err*100:.4f}%  ({ok}/{total} legal)",
          flush=True)

    res_dir = cfg["paths"]["results_dir"]
    os.makedirs(res_dir, exist_ok=True)
    out = os.path.join(res_dir, f"gpt_{profile}{suffix}_eval.json")
    with open(out, "w") as f:
        json.dump({"profile": profile, "arch": args.arch,
                   "expansion": args.expansion if args.arch == "bilinear" else None,
                   "legal_move_error": err,
                   "legal": ok, "total": total, "ckpt": ckpt_path}, f, indent=2)
    print(f"[train] wrote {out}", flush=True)


if __name__ == "__main__":
    main()
