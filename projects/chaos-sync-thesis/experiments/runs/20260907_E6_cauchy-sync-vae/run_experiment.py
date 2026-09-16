"""固定splitでCauchy潜在・tan力学・同期を段階比較し、再現資料を保存する。"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import platform
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import scipy
import sklearn
import torch
from PIL import Image
from scipy.stats import cauchy, kstest
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from torch import Tensor, nn
from torch.nn import functional as F

CONFIG = dict(split_seed=607, seeds=[607, 608, 609], latent_dim=8,
              epochs=40, batch_size=128, train_samples=4, val_samples=16,
              test_samples=64, learning_rate=0.001, beta=1.5, steps=4,
              dtype="float64", kl_weight=1.0, binarization_threshold=0.5)
VARIANTS = [("gaussian", "normal", 0, 0.0), ("cauchy", "cauchy", 0, 0.0),
            ("chaos", "cauchy", 4, 0.0), ("sync_025", "cauchy", 4, 0.25),
            ("sync_045", "cauchy", 4, 0.45), ("sync_050", "cauchy", 4, 0.5)]


def save_json(path: Path, value: Any) -> None:
    """非有限値を成功した数値として保存せず、UTF-8 JSONを書く。"""
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8", newline="\n")


def finite(value: Tensor) -> Tensor:
    """力学や勾配の非有限値を検出して、置換せず停止する。"""
    if not torch.isfinite(value).all():
        raise FloatingPointError("Non-finite primary-path value")
    return value


def evolve(z: Tensor, steps: int, coupling: float, beta: float = 1.5) -> Tensor:
    """末尾8座標を2ノード組にし、同期多様体を保つtan出力拡散を反復する。"""
    if z.shape[-1] % 2 or steps < 0 or not 0 <= coupling <= 0.5:
        raise ValueError("Even latent dimension, nonnegative steps, coupling in [0,.5] required")
    state = z.reshape(*z.shape[:-1], -1, 2)
    for _ in range(steps):
        mapped = finite(torch.tan(beta * state))
        state = finite((1 - coupling) * mapped + coupling * mapped.flip(-1))
    return state.reshape(z.shape)


def cauchy_kl(loc: Tensor, scale: Tensor) -> Tensor:
    """標準Cauchyへの積分布KLを閉形式で計算する（潜在座標和）。"""
    return torch.log(((scale + 1).square() + loc.square()) / (4 * scale)).sum(-1)


class VAE(nn.Module):
    """初期潜在へKLを課し、固定力学後の有界観測だけで復号するVAE。"""

    def __init__(self, family: str, steps: int, coupling: float, dim: int = 8) -> None:
        super().__init__()
        if family not in ("normal", "cauchy") or dim % 2:
            raise ValueError("Unknown posterior family or odd latent dimension")
        self.family, self.steps, self.coupling, self.dim = family, steps, coupling, dim
        self.encoder = nn.Sequential(nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, 2 * dim))
        self.decoder = nn.Sequential(nn.Linear(dim, 64), nn.Tanh(), nn.Linear(64, 64))

    def posterior(self, x: Tensor) -> Tuple[Any, Tensor]:
        """有界位置・正尺度から事後と解析KLを返す。"""
        raw_loc, raw_scale = self.encoder(x).chunk(2, -1)
        loc, scale = 3 * raw_loc.tanh(), 0.1 + 1.9 * raw_scale.sigmoid()
        if self.family == "normal":
            q = torch.distributions.Normal(loc, scale)
            kl = (0.5 * (loc.square() + scale.square() - 1) - scale.log()).sum(-1)
        else:
            q = torch.distributions.Cauchy(loc, scale)
            kl = cauchy_kl(loc, scale)
        return q, kl

    def decode(self, z: Tensor) -> Tuple[Tensor, Tensor]:
        """生成時・再構成時で共通の力学、観測、制約付きlogitsを返す。"""
        state = evolve(z, self.steps, self.coupling)
        features = 2 / math.pi * state.atan()
        logits = 8 * (self.decoder(features) / 8).tanh()
        return finite(logits), state

    def objective(self, x: Tensor, samples: int) -> Tensor:
        """独立sample間LOO baselineにより、力学を微分せずELBO勾配を推定する。"""
        if samples < 2:
            raise ValueError("Leave-one-out baseline requires at least two samples")
        q, kl = self.posterior(x)
        # tan軌道のpathwise微分は極近傍で期待値と微分の交換を保証できない。
        z = finite(q.sample((samples,)))
        logits, _ = self.decode(z)
        costs = F.binary_cross_entropy_with_logits(logits, x.expand_as(logits), reduction="none").sum(-1)
        baseline = (costs.detach().sum(0, keepdim=True) - costs.detach()) / (samples - 1)
        score = ((costs.detach() - baseline) * q.log_prob(z).sum(-1)).mean()
        return costs.mean() + kl.mean() + score


@torch.no_grad()
def evaluate(model: VAE, x: Tensor, samples: int, seed: int) -> Dict[str, np.ndarray]:
    """未学習データの負ELBO、再構成分布、有界同期指標を保存可能な配列で返す。"""
    model.eval()
    collected: Dict[str, List[np.ndarray]] = {key: [] for key in
        ("nll", "kl", "negative_elbo", "probabilities", "bounded_sync", "exact_sync")}
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        for batch in x.split(128):
            q, kl = model.posterior(batch)
            logits, state = model.decode(q.sample((samples,)))
            nll = F.binary_cross_entropy_with_logits(logits, batch.expand_as(logits), reduction="none").sum(-1).mean(0)
            pairs = state.reshape(samples, len(batch), -1, 2)
            sync = (pairs[..., 0].atan() - pairs[..., 1].atan()).abs().mean((0, 2)) / math.pi
            exact = (pairs[..., 0] == pairs[..., 1]).double().mean((0, 2))
            vals = [nll, kl, nll + kl, logits.sigmoid().mean(0), sync, exact]
            for key, value in zip(collected, vals):
                collected[key].append(finite(value).cpu().numpy())
    return {key: np.concatenate(value) for key, value in collected.items()}


def closure_check() -> List[Dict[str, Any]]:
    """非結合一段のlocation-scale閉包を独立iid集合のCDFで検証する。"""
    rng = np.random.default_rng(60700)
    records = []
    n, count, alpha = 50000, 9, 0.01
    threshold = math.sqrt(math.log(2 * count / alpha) / (2 * n))
    for loc in [-0.7, 0.0, 0.8]:
        for scale in [0.2, 0.8, 1.5]:
            pole = np.tan(1.5 * complex(loc, scale))
            source = loc + scale * rng.standard_cauchy(n)
            output = np.tan(1.5 * source)
            if not np.isfinite(output).all():
                raise FloatingPointError("Non-finite closure sample")
            ks = float(kstest(output, cauchy.cdf, args=(pole.real, pole.imag)).statistic)
            records.append(dict(loc=loc, scale=scale, next_loc=pole.real,
                                next_scale=pole.imag, ks=ks, dkw_threshold=threshold, passed=ks <= threshold))
    return records


def contact_sheet(probabilities: np.ndarray, path: Path) -> None:
    """保存した8x8画素配列の先頭64枚を、画質評価用の拡大一覧にする。"""
    tiles = np.rint(255 * probabilities[:64].reshape(8, 8, 8, 8)).astype(np.uint8)
    sheet = tiles.transpose(0, 2, 1, 3).reshape(64, 64)
    Image.fromarray(sheet).resize((512, 512), Image.Resampling.NEAREST).save(path)


def run(output: Path) -> None:
    """事前設定に従って学習し、split、checkpoint、全test指標、事前生成、hashを残す。"""
    output.mkdir(parents=True, exist_ok=False)
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    save_json(output / "config.json", dict(CONFIG, variants=VARIANTS))
    save_json(output / "environment.json", dict(python=platform.python_version(), platform=platform.platform(),
              torch=torch.__version__, numpy=np.__version__, scipy=scipy.__version__, sklearn=sklearn.__version__))
    save_json(output / "status.json", dict(status="running"))
    digits = load_digits()
    x = (digits.data / 16 >= CONFIG["binarization_threshold"]).astype(np.float64)
    train_idx, rest = train_test_split(np.arange(len(x)), test_size=0.4, stratify=digits.target, random_state=607)
    val_idx, test_idx = train_test_split(rest, test_size=0.5, stratify=digits.target[rest], random_state=608)
    np.savez_compressed(output / "data_split.npz", raw=digits.data, x=x, labels=digits.target,
                        train=train_idx, validation=val_idx, test=test_idx)
    train, val, test = [torch.from_numpy(x[idx]) for idx in (train_idx, val_idx, test_idx)]
    save_json(output / "closure.json", closure_check())
    rows = []
    # prior-only decoderの比較基準。画素確率はtrainだけで推定する。
    null_prob = np.clip(x[train_idx].mean(0), 1e-4, 1 - 1e-4)
    null_nll = -(x[test_idx] * np.log(null_prob) + (1 - x[test_idx]) * np.log1p(-null_prob)).sum(-1)
    np.save(output / "null_test_nll.npy", null_nll)
    for seed in CONFIG["seeds"]:
        for name, family, steps, coupling in VARIANTS:
            start = time.perf_counter()
            torch.manual_seed(seed)
            model = VAE(family, steps, coupling).double()
            optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["learning_rate"])
            best_value, best_epoch, best_state = math.inf, 0, None
            history = []
            order_rng = torch.Generator().manual_seed(seed + 100)
            for epoch in range(CONFIG["epochs"]):
                model.train()
                for idx in torch.randperm(len(train), generator=order_rng).split(CONFIG["batch_size"]):
                    optimizer.zero_grad()
                    loss = finite(model.objective(train[idx], CONFIG["train_samples"]))
                    loss.backward()
                    for parameter in model.parameters():
                        if parameter.grad is not None:
                            finite(parameter.grad)
                    optimizer.step()
                scores = evaluate(model, val, CONFIG["val_samples"], seed + 10000)
                value = float(scores["negative_elbo"].mean())
                history.append(dict(epoch=epoch + 1, validation_negative_elbo=value,
                                    nll=float(scores["nll"].mean()), kl=float(scores["kl"].mean())))
                if value < best_value:
                    best_value, best_epoch, best_state = value, epoch + 1, copy.deepcopy(model.state_dict())
            assert best_state is not None
            model.load_state_dict(best_state)
            scores = evaluate(model, test, CONFIG["test_samples"], seed + 20000)
            with torch.no_grad(), torch.random.fork_rng():
                torch.manual_seed(seed + 30000)
                prior = (torch.distributions.Normal(0., 1.) if family == "normal"
                         else torch.distributions.Cauchy(0., 1.))
                z = prior.sample((256, model.dim))
                logits, state = model.decode(z)
                probabilities = logits.sigmoid().numpy()
                images = torch.bernoulli(logits.sigmoid()).numpy()
            prefix = f"{name}_seed{seed}"
            np.savez_compressed(output / f"{prefix}.npz", **scores, prior_z=z.numpy(),
                                prior_terminal=state.numpy(), prior_probabilities=probabilities, prior_images=images)
            torch.save(best_state, output / f"{prefix}.pt")
            save_json(output / f"{prefix}_history.json", history)
            if seed == CONFIG["seeds"][0]:
                contact_sheet(probabilities, output / f"{name}_prior_probabilities.png")
                contact_sheet(images, output / f"{name}_prior_samples.png")
                contact_sheet(scores["probabilities"], output / f"{name}_reconstructions.png")
            row = dict(variant=name, seed=seed, best_epoch=best_epoch, validation_negative_elbo=best_value,
                       parameter_count=sum(p.numel() for p in model.parameters()),
                       test_nll=float(scores["nll"].mean()), test_kl=float(scores["kl"].mean()),
                       test_negative_elbo=float(scores["negative_elbo"].mean()),
                       test_bounded_sync=float(scores["bounded_sync"].mean()),
                       test_exact_sync=float(scores["exact_sync"].mean()), seconds=time.perf_counter() - start)
            rows.append(row)
            save_json(output / "metrics.json", dict(rows=rows, null_test_nll=float(null_nll.mean())))
            print(json.dumps(row), flush=True)
    contact_sheet(x[test_idx], output / "test_targets.png")
    save_json(output / "status.json", dict(status="completed", scientific_status="pilot_only"))
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in output.iterdir() if p.is_file()}
    hashes["../run_experiment.py"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    save_json(output / "sha256.json", hashes)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "artifacts")
    args = parser.parse_args()
    run(args.output)
