"""Shared helpers for the Othello-GPT reproduction.

Centralises: config loading, the canonical 61-token vocabulary (so the pilot
model, the full model, and the authors' released checkpoint all share one
stoi/itos mapping), pickle-shard I/O, and a thin CharDataset with a *fixed*
vocab.

The upstream `mingpt`/`data` packages are imported by inserting the vendored
`upstream/` directory onto sys.path (see `add_upstream_to_path`).
"""
import os
import sys
import glob
import pickle
import itertools

import yaml

# --- canonical Othello vocabulary -------------------------------------------
# Othello is 8x8 = 64 tiles; the 4 centre tiles (d4,d5,e4,e5 -> 27,28,35,36)
# are the fixed opening and are never played, leaving 60 legal move tokens.
# CharDataset builds chars = sorted(set(moves) + [-100]); -100 (pad / CE
# ignore) sorts to the front -> index 0, which matches GPT's loss
# `ignore_index=0`.  We pin this mapping so it is identical regardless of which
# games happen to appear in a (possibly small) data sample.
CENTER_TILES = {27, 28, 35, 36}
LEGAL_TILES = [t for t in range(64) if t not in CENTER_TILES]   # 60 tiles
CANONICAL_CHARS = sorted([-100] + LEGAL_TILES)                  # 61 entries
STOI = {ch: i for i, ch in enumerate(CANONICAL_CHARS)}
ITOS = {i: ch for i, ch in enumerate(CANONICAL_CHARS)}
VOCAB_SIZE = len(CANONICAL_CHARS)   # 61
MAX_LEN = 60                        # longest possible Othello game
BLOCK_SIZE = MAX_LEN - 1           # 59


def add_upstream_to_path(cfg):
    """Put the vendored upstream repo on sys.path so `import mingpt`, `import
    data` resolve to the original authors' code."""
    up = cfg["paths"]["upstream"]
    if up not in sys.path:
        sys.path.insert(0, up)


def load_config(path=None):
    if path is None:
        here = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(here, os.pardir, "configs", "config.yaml")
    with open(path) as f:
        cfg = yaml.safe_load(f)
    return cfg


def profile_get(cfg, section, key):
    """Read a value that may be overridden per-profile.

    e.g. train.batch_size lives under train.<profile>.batch_size, while
    train.learning_rate is profile-independent."""
    prof = cfg["profile"]
    sec = cfg[section]
    if prof in sec and isinstance(sec[prof], dict) and key in sec[prof]:
        return sec[prof][key]
    return sec[key]


# --- shard I/O ---------------------------------------------------------------
def shard_path(data_dir, profile, idx):
    return os.path.join(data_dir, f"{profile}_shard_{idx:04d}.pkl")


def write_shard(games, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(games, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_all_games(data_dir, profile):
    """Load every shard for a profile, dedup, return a single list of games."""
    paths = sorted(glob.glob(os.path.join(data_dir, f"{profile}_shard_*.pkl")))
    if not paths:
        raise FileNotFoundError(
            f"No shards found at {data_dir} for profile '{profile}'. Run gen_data first.")
    games = []
    for p in paths:
        with open(p, "rb") as f:
            games.extend(pickle.load(f))
    # dedup (games are lists -> sort+groupby on tuples)
    games.sort()
    games = [k for k, _ in itertools.groupby(games)]
    return games


def split_train_val(games, val_games):
    """Deterministic train/val split (games already deduped + sorted, so this
    is reproducible)."""
    val = games[:val_games]
    train = games[val_games:]
    return train, val


def val_cache_path(cfg, profile):
    return os.path.join(cfg["paths"]["ckpt_dir"], f"val_games_{profile}.pkl")


def save_val_cache(cfg, profile, val_games):
    p = val_cache_path(cfg, profile)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "wb") as f:
        pickle.dump(val_games, f, protocol=pickle.HIGHEST_PROTOCOL)


def get_val_games(cfg, profile):
    """Return the held-out val games. Reads the cache written by train_gpt if
    present (cheap); otherwise reconstructs it from the shards (and caches it).
    Avoids loading all 20M games in the probe/intervene jobs at full scale."""
    p = val_cache_path(cfg, profile)
    if os.path.exists(p):
        with open(p, "rb") as f:
            return pickle.load(f)
    games = load_all_games(cfg["paths"]["data_dir"], profile)
    _, val = split_train_val(games, cfg["data"]["val_games"])
    save_val_cache(cfg, profile, val)
    return val


# --- fixed-vocab dataset -----------------------------------------------------
def make_dataset(games):
    """Build a CharDataset (from upstream) but overwrite its vocab with the
    canonical mapping so every run/checkpoint is mutually compatible."""
    from mingpt.dataset import CharDataset
    ds = CharDataset(games)
    ds.stoi = dict(STOI)
    ds.itos = dict(ITOS)
    ds.vocab_size = VOCAB_SIZE
    ds.max_len = MAX_LEN
    ds.block_size = BLOCK_SIZE
    return ds
