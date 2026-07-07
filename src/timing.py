"""Measure training throughput to size the full run against the 48h GPU cap.

Times forward+backward steps on real data at a given batch size, reports
iters/sec, then projects per-epoch and 250-epoch time for the full (20M) set.
"""
import os, time, argparse
import torch
import common


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--batch_size", type=int, default=4096)
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--full_games", type=int, default=20_000_000)
    args = ap.parse_args()
    cfg = common.load_config(args.config)
    common.add_upstream_to_path(cfg)
    from mingpt.model import GPT, GPTConfig
    from torch.utils.data.dataloader import DataLoader

    device = "cuda"
    ngpu = torch.cuda.device_count()
    print(f"[timing] GPUs visible: {ngpu}", flush=True)
    for i in range(ngpu):
        print("   ", torch.cuda.get_device_name(i), flush=True)

    games = common.load_all_games(cfg["paths"]["data_dir"], "pilot")
    ds = common.make_dataset(games)
    mconf = GPTConfig(common.VOCAB_SIZE, common.BLOCK_SIZE, n_layer=8, n_head=8, n_embd=512)
    model = GPT(mconf)
    if ngpu > 1:
        model = torch.nn.DataParallel(model)
    model = model.to(device)
    raw = model.module if hasattr(model, "module") else model
    opt = torch.optim.AdamW(raw.parameters(), lr=5e-4)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                        num_workers=8, pin_memory=True, drop_last=True)

    model.train()
    it = iter(loader)
    # warmup
    for _ in range(3):
        x, y = next(it); x, y = x.to(device), y.to(device)
        _, loss = model(x, y); loss.mean().backward(); opt.step(); opt.zero_grad()
    torch.cuda.synchronize()
    t0 = time.time(); n = 0
    for _ in range(args.steps):
        try:
            x, y = next(it)
        except StopIteration:
            it = iter(loader); x, y = next(it)
        x, y = x.to(device), y.to(device)
        _, loss = model(x, y); loss.mean().backward(); opt.step(); opt.zero_grad()
        n += 1
    torch.cuda.synchronize()
    dt = time.time() - t0
    ips = n / dt
    iters_per_epoch = args.full_games / args.batch_size
    sec_per_epoch = iters_per_epoch / ips
    print(f"\n[timing] batch={args.batch_size} gpus={ngpu}: {ips:.2f} it/s "
          f"({dt/n*1000:.0f} ms/it)", flush=True)
    print(f"[timing] full 20M: {iters_per_epoch:.0f} it/epoch -> "
          f"{sec_per_epoch/60:.1f} min/epoch", flush=True)
    print(f"[timing] 250 epochs -> {sec_per_epoch*250/3600:.1f} h "
          f"(48h cap fits ~{48*3600/sec_per_epoch:.0f} epochs)", flush=True)


if __name__ == "__main__":
    main()
