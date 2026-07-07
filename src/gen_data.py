"""Generate the synthetic Othello dataset.

Faithful to the paper: each game is produced by `get_ood_game` from the
vendored `data/othello.py`, i.e. uniformly random *legal* play until no move
remains (a leaf of the game tree). Games are written as pickle shards of
move-index lists under paths.data_dir.

Runs under Slurm on a CPU node (uses all allocated cores). Example:
    python gen_data.py --config ../configs/config.yaml --profile pilot
"""
import os
import time
import random
import argparse
import multiprocessing as mp

from tqdm import tqdm

import common


def _gen_one(seed):
    # distinct RNG state per task so workers don't generate identical games
    random.seed(seed)
    from data.othello import get_ood_game
    return get_ood_game(0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--profile", default=None, help="override cfg.profile")
    args = ap.parse_args()

    cfg = common.load_config(args.config)
    if args.profile:
        cfg["profile"] = args.profile
    common.add_upstream_to_path(cfg)

    profile = cfg["profile"]
    data_dir = cfg["paths"]["data_dir"]
    shard_size = cfg["data"]["shard_size"]
    base_seed = cfg["data"]["seed"]
    n_games = common.profile_get(cfg, "data", "n_games")

    os.makedirs(data_dir, exist_ok=True)
    ncpu = int(os.environ.get("SLURM_CPUS_PER_TASK", mp.cpu_count()))
    print(f"[gen_data] profile={profile} n_games={n_games} shard_size={shard_size} "
          f"cpus={ncpu} out={data_dir}", flush=True)

    t0 = time.time()
    n_shards = (n_games + shard_size - 1) // shard_size
    total_written = 0
    with mp.Pool(ncpu) as pool:
        for s in range(n_shards):
            lo = s * shard_size
            hi = min((s + 1) * shard_size, n_games)
            # unique seeds across the whole run
            seeds = range(base_seed + lo, base_seed + hi)
            games = []
            for g in tqdm(pool.imap_unordered(_gen_one, seeds, chunksize=64),
                          total=hi - lo, desc=f"shard {s+1}/{n_shards}"):
                games.append(g)
            # local dedup (cheap; global dedup happens at load time)
            games.sort()
            import itertools
            games = [k for k, _ in itertools.groupby(games)]
            path = common.shard_path(data_dir, profile, s)
            common.write_shard(games, path)
            total_written += len(games)
            print(f"[gen_data] wrote {path}: {len(games)} unique games "
                  f"({total_written} total, {time.time()-t0:.0f}s)", flush=True)

    print(f"[gen_data] DONE: {total_written} games in {n_shards} shards, "
          f"{time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
