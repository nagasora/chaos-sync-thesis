"""保存済みE6の分割・選択・数値・同期・ハッシュを学習せずに監査する。"""
from pathlib import Path
import hashlib
import json
from typing import Dict

import numpy as np
import torch


def validate(path: Path) -> Dict[str, bool]:
    """学習結果を再計算せず、独立した保存物間の契約を検証する。"""
    checks = {}
    hashes = json.loads((path / "sha256.json").read_text())
    checks["hashes"] = all(hashlib.sha256((path / name).read_bytes()).hexdigest() == digest
                          for name, digest in hashes.items())
    data = np.load(path / "data_split.npz")
    splits = [set(data[k].tolist()) for k in ("train", "validation", "test")]
    checks["split"] = (all(not splits[i] & splits[j] for i in range(3) for j in range(i))
                       and len(set.union(*splits)) == len(data["x"]))
    checks["preprocessing"] = np.array_equal(data["x"], (data["raw"] / 16 >= .5).astype(float))
    rows = json.loads((path / "metrics.json").read_text())["rows"]
    checks["conditions"] = len(rows) == len({(r["variant"], r["seed"]) for r in rows}) == 18
    checks["parameter_budget"] = {r["parameter_count"] for r in rows} == {9936}
    checks["completed"] = json.loads((path / "status.json").read_text())["status"] == "completed"
    for row in rows:
        prefix = f'{row["variant"]}_seed{row["seed"]}'
        arrays = np.load(path / f"{prefix}.npz")
        for key in arrays.files:
            checks[f"{prefix}_{key}_finite"] = bool(np.isfinite(arrays[key]).all())
        checks[prefix + "_elbo"] = bool(np.allclose(arrays["negative_elbo"], arrays["nll"] + arrays["kl"], atol=1e-12))
        checks[prefix + "_metrics"] = all(abs(float(arrays[k].mean()) - row["test_" + k]) < 1e-12
                                           for k in ("nll", "kl", "negative_elbo", "bounded_sync", "exact_sync"))
        history = json.loads((path / f"{prefix}_history.json").read_text())
        checks[prefix + "_selection"] = row["best_epoch"] == min(history, key=lambda h: h["validation_negative_elbo"])["epoch"]
        checks[prefix + "_binary"] = bool(np.isin(arrays["prior_images"], [0, 1]).all())
        if row["variant"] == "sync_050":
            pairs = arrays["prior_terminal"].reshape(-1, 4, 2)
            checks[prefix + "_sync"] = bool(np.array_equal(pairs[..., 0], pairs[..., 1]) and (arrays["exact_sync"] == 1).all())
        torch.load(path / f"{prefix}.pt", weights_only=True)
    checks["closure"] = all(r["passed"] for r in json.loads((path / "closure.json").read_text()))
    return checks


if __name__ == "__main__":
    directory = Path(__file__).parent / "artifacts"
    results = validate(directory)
    report = dict(passed=sum(results.values()), total=len(results), checks=results,
                  note="Artifact integrity audit, not independent replication of scientific performance")
    (directory / "validation.json").write_text(json.dumps(report, indent=2), encoding="utf-8", newline="\n")
    if not all(results.values()):
        raise AssertionError([key for key, value in results.items() if not value])
    print(f'{report["passed"]}/{report["total"]} passed')
