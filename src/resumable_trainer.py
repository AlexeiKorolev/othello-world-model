"""Checkpoint/resume trainer so a long run can be split across time-limited jobs.

Reimplements the loop of `mingpt.trainer.Trainer.train()` verbatim (same AdamW via
`configure_optimizers`, same token-based linear-warmup + cosine LR decay, same
grad-norm clip), and adds:

  * a full-state checkpoint each epoch -- model + optimizer + LR token counter +
    epoch index + RNG state -- written atomically to `<ckpt>.state`, plus the
    model-only `<ckpt>` that downstream jobs load;
  * `train(wall_budget_sec)`: after each epoch, if the *next* epoch (estimated
    from measured epoch time, +20% margin) would exceed the budget, pause and
    return False. Returns True only when all `max_epochs` are done;
  * bf16 autocast + `torch.compile` on the training forward pass (both on by
    default when CUDA is available -- this workload is compute-bound, not
    memory-bound, per bench_speed.py's H200 sweep, so fp32 was leaving real
    throughput on the table). Only the hot forward call goes through the
    compiled/autocast path (`self.train_model`); checkpointing, `_raw()`, and
    `configure_optimizers` all still see the plain uncompiled model
    (`self.model`), since compile/DataParallel wrapping only needs to touch
    the loop that calls the model, not anything that introspects or saves it.
    DataParallel is skipped entirely on a single visible GPU (the common case
    here) since it's a no-op wrapper in that case and one less thing for
    compile to wrap.

Because the LR schedule is driven by the restored `self.tokens`, chunked training
follows the identical LR curve as one continuous run.
"""
import os
import math
import time

import numpy as np
import torch
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm


def _state_path(ckpt_path):
    return ckpt_path + ".state"


def _atomic_save(obj, path):
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


class ResumableTrainer:
    def __init__(self, model, train_dataset, config, use_amp=True, compile_model=True):
        self.model = model
        self.train_dataset = train_dataset
        self.config = config
        self.device = "cpu"
        self.amp_device_type = "cpu"
        self.use_amp = False
        self.train_model = self.model
        if torch.cuda.is_available():
            self.device = torch.device(f"cuda:{torch.cuda.current_device()}")
            self.amp_device_type = "cuda"
            self.use_amp = use_amp
            self.model = self.model.to(self.device)
            if torch.cuda.device_count() > 1:
                self.model = torch.nn.DataParallel(self.model)
            self.train_model = (torch.compile(self.model, dynamic=False)
                                if compile_model else self.model)
        self.tokens = 0
        self.start_epoch = 0

    def _raw(self):
        return self.model.module if hasattr(self.model, "module") else self.model

    def save_state(self, optimizer, completed_epochs):
        raw = self._raw()
        state = {
            "model": raw.state_dict(),
            "optimizer": optimizer.state_dict(),
            "tokens": int(self.tokens),
            "epoch": completed_epochs,            # number of epochs finished
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": (torch.cuda.get_rng_state_all()
                         if torch.cuda.is_available() else None),
        }
        _atomic_save(state, _state_path(self.config.ckpt_path))
        _atomic_save(raw.state_dict(), self.config.ckpt_path)

    def maybe_resume(self, optimizer):
        sp = _state_path(self.config.ckpt_path)
        if not os.path.exists(sp):
            return
        st = torch.load(sp, map_location=self.device)
        self._raw().load_state_dict(st["model"])
        optimizer.load_state_dict(st["optimizer"])
        self.tokens = st["tokens"]
        self.start_epoch = st["epoch"]
        try:
            rng = st["torch_rng"]
            torch.set_rng_state(rng.cpu() if hasattr(rng, "cpu") else rng)
            if st.get("cuda_rng") is not None and torch.cuda.is_available():
                torch.cuda.set_rng_state_all(list(st["cuda_rng"]))
        except Exception as e:
            print(f"[resume] RNG restore skipped: {e}", flush=True)
        print(f"[resume] resumed at epoch {self.start_epoch}/{self.config.max_epochs}, "
              f"tokens {self.tokens}", flush=True)

    def _run_epoch(self, optimizer, epoch):
        config = self.config
        self.model.train(True)
        loader = DataLoader(self.train_dataset, shuffle=True, pin_memory=True,
                            batch_size=config.batch_size, num_workers=config.num_workers)
        pbar = tqdm(enumerate(loader), total=len(loader), mininterval=30)
        for it, (x, y) in pbar:
            x = x.to(self.device); y = y.to(self.device)
            with torch.set_grad_enabled(True), torch.autocast(
                    device_type=self.amp_device_type, dtype=torch.bfloat16,
                    enabled=self.use_amp):
                _, loss = self.train_model(x, y)
                loss = loss.mean()
            self.model.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), config.grad_norm_clip)
            optimizer.step()
            if config.lr_decay:
                self.tokens += (y >= 0).sum()
                if self.tokens < config.warmup_tokens:
                    lr_mult = float(self.tokens) / float(max(1, config.warmup_tokens))
                else:
                    progress = (float(self.tokens - config.warmup_tokens)
                                / float(max(1, config.final_tokens - config.warmup_tokens)))
                    lr_mult = max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))
                lr = config.learning_rate * lr_mult
                for g in optimizer.param_groups:
                    g["lr"] = lr
            else:
                lr = config.learning_rate
            pbar.set_description(f"epoch {epoch+1} iter {it}: loss {loss.item():.5f} lr {lr:e}")

    def train(self, wall_budget_sec=None):
        raw = self._raw()
        optimizer = raw.configure_optimizers(self.config)
        self.maybe_resume(optimizer)

        t0 = time.time()
        ran = 0
        epoch = self.start_epoch
        while epoch < self.config.max_epochs:
            self._run_epoch(optimizer, epoch)
            epoch += 1
            ran += 1
            self.save_state(optimizer, epoch)
            print(f"[resume] completed epoch {epoch}/{self.config.max_epochs}; state saved",
                  flush=True)
            if epoch >= self.config.max_epochs:
                return True
            if wall_budget_sec is not None:
                elapsed = time.time() - t0
                per_epoch = elapsed / ran
                if elapsed + 1.2 * per_epoch > wall_budget_sec:
                    print(f"[resume] wall budget reached (elapsed {elapsed/3600:.2f}h, "
                          f"~{per_epoch/3600:.2f}h/epoch); pausing after epoch {epoch}",
                          flush=True)
                    return False
        return True
