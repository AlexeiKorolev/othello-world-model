"""Throughput benchmark: sweep batch size on a single GPU to find the largest
batch that fits in memory and the batch size that maximizes training
throughput (tokens/sec), for the normal (GELU) Othello-GPT.

Meant for GPUs reached via the free/preemptible ckpt-all queue (e.g. H200):
those jobs can be preempted and requeued (QOS PreemptMode=REQUEUE), so this
persists a small state file after every batch size measured -- a requeued run
loads it, skips batch sizes already measured, and continues the sweep rather
than starting over.

Measures real training steps (forward + backward + grad-clip + AdamW step) on
a fixed batch reused for every timed iteration, so dataloader overhead doesn't
skew the numbers -- this isolates GPU compute throughput specifically.

    python bench_speed.py --config ../configs/config.yaml --profile pilot --tag h200
"""
import os
import json
import time
import argparse

import torch

import common


def state_path(res_dir, tag):
    return os.path.join(res_dir, f"bench_{tag}_state.json")


def load_state(path):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {"tag": None, "results": []}


def save_state(path, state):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(state, f, indent=2)
    os.replace(tmp, path)


def bench_one(model, train_model, optimizer, x, y, grad_norm_clip, device, use_amp,
             warmup, iters):
    """model = plain module (params/grad-clip); train_model = the (possibly
    torch.compile'd) callable used for the forward pass -- same split as
    resumable_trainer.ResumableTrainer, so this measures exactly what real
    training will run."""
    model.train()
    def step():
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
            _, loss = train_model(x, y)
            loss = loss.mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_norm_clip)
        optimizer.step()

    for _ in range(warmup):
        step()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    for _ in range(iters):
        step()
    torch.cuda.synchronize()
    dt = time.time() - t0
    peak_mem = torch.cuda.max_memory_allocated(device) / 1e9
    return dt / iters, peak_mem


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--profile", default=None)
    ap.add_argument("--tag", default="gpu", help="label for the GPU/run, used in output filenames")
    ap.add_argument("--batch_sizes",
                    default="512,1024,2048,4096,8192,16384,32768,65536,131072",
                    help="comma-separated batch sizes to try, ascending")
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--iters", type=int, default=20)
    args = ap.parse_args()

    cfg = common.load_config(args.config)
    if args.profile:
        cfg["profile"] = args.profile
    common.add_upstream_to_path(cfg)
    profile = cfg["profile"]

    assert torch.cuda.is_available(), "bench_speed.py needs a GPU"
    device = "cuda"
    gpu_name = torch.cuda.get_device_name(0)
    total_mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"[bench] GPU: {gpu_name}  ({total_mem_gb:.0f} GB)", flush=True)

    from mingpt.model import GPT, GPTConfig
    from mingpt.trainer import TrainerConfig

    games = common.load_all_games(cfg["paths"]["data_dir"], profile)
    train_games, _val = common.split_train_val(games, cfg["data"]["val_games"])
    train_dataset = common.make_dataset(train_games)
    block = train_dataset.block_size

    mconf = GPTConfig(common.VOCAB_SIZE, common.BLOCK_SIZE,
                      n_layer=cfg["model"]["n_layer"], n_head=cfg["model"]["n_head"],
                      n_embd=cfg["model"]["n_embd"])

    res_dir = cfg["paths"]["results_dir"]
    os.makedirs(res_dir, exist_ok=True)
    spath = state_path(res_dir, args.tag)
    state = load_state(spath)
    state["tag"] = args.tag
    state["profile"] = profile
    state["gpu_name"] = gpu_name
    state["gpu_mem_gb"] = round(total_mem_gb, 1)
    done_bs = {r["batch_size"] for r in state["results"]}

    candidates = [int(b) for b in args.batch_sizes.split(",")]
    max_bs = len(train_dataset)
    grad_norm_clip = cfg["train"]["grad_norm_clip"]

    for bs in candidates:
        if bs in done_bs:
            print(f"[bench] batch_size={bs}: already measured (resumed), skipping", flush=True)
            continue
        if bs > max_bs:
            print(f"[bench] batch_size={bs} exceeds dataset size ({max_bs}), stopping sweep",
                  flush=True)
            break
        if any(r.get("oom") for r in state["results"]):
            print(f"[bench] earlier batch size OOM'd; stopping sweep before {bs}", flush=True)
            break

        model = GPT(mconf).to(device)
        train_model = torch.compile(model, dynamic=False)
        tconf = TrainerConfig(learning_rate=cfg["train"]["learning_rate"],
                              betas=tuple(cfg["train"]["betas"]),
                              weight_decay=cfg["train"]["weight_decay"])
        optimizer = model.configure_optimizers(tconf)

        ld = torch.utils.data.DataLoader(train_dataset, batch_size=bs, shuffle=True)
        x, y = next(iter(ld))
        x, y = x.to(device), y.to(device)

        try:
            sec_per_iter, peak_mem = bench_one(model, train_model, optimizer, x, y,
                                               grad_norm_clip, device, True,
                                               args.warmup, args.iters)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"[bench] batch_size={bs}: OOM ({e})", flush=True)
                state["results"].append({"batch_size": bs, "oom": True})
                save_state(spath, state)
                del model, train_model, optimizer
                torch.cuda.empty_cache()
                break
            raise
        del model, train_model, optimizer
        torch.cuda.empty_cache()

        samples_per_sec = bs / sec_per_iter
        tokens_per_sec = samples_per_sec * block
        epoch_sec_est = len(train_dataset) / samples_per_sec
        record = {"batch_size": bs, "oom": False,
                  "sec_per_iter": round(sec_per_iter, 4),
                  "samples_per_sec": round(samples_per_sec, 1),
                  "tokens_per_sec": round(tokens_per_sec, 1),
                  "peak_mem_gb": round(peak_mem, 2),
                  "est_epoch_sec": round(epoch_sec_est, 1)}
        state["results"].append(record)
        save_state(spath, state)
        print(f"[bench] batch_size={bs}: {sec_per_iter*1000:.1f} ms/iter  "
              f"{samples_per_sec:.0f} samples/s  {tokens_per_sec:.0f} tok/s  "
              f"peak {peak_mem:.1f} GB  ~{epoch_sec_est:.0f}s/epoch", flush=True)

    ok = [r for r in state["results"] if not r.get("oom")]
    if ok:
        best = max(ok, key=lambda r: r["samples_per_sec"])
        largest = max(ok, key=lambda r: r["batch_size"])
        state["summary"] = {
            "best_throughput_batch_size": best["batch_size"],
            "best_samples_per_sec": best["samples_per_sec"],
            "best_tokens_per_sec": best["tokens_per_sec"],
            "largest_working_batch_size": largest["batch_size"],
            "largest_batch_peak_mem_gb": largest["peak_mem_gb"],
        }
        print(f"[bench] DONE. Best throughput: batch_size={best['batch_size']} "
              f"({best['samples_per_sec']:.0f} samples/s, {best['tokens_per_sec']:.0f} tok/s). "
              f"Largest batch that fit: {largest['batch_size']} "
              f"({largest['peak_mem_gb']:.1f} GB peak).", flush=True)

    out = os.path.join(res_dir, f"bench_{args.tag}.json")
    save_state(out, state)
    print(f"[bench] wrote {out}", flush=True)


if __name__ == "__main__":
    main()
