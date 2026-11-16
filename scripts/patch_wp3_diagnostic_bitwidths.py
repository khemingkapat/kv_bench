#!/usr/bin/env python3
"""Patch Option B physical effective bitwidths in existing LongMemEval JSONs.

Recomputes ``diag_mean_overall_effective_bitwidth``,
``diag_mean_min_session_effective_bitwidth``, and
``diag_mean_compression_ratio`` for quantization techniques whose
``overall_retention_ratio`` is 1.00 and whose scalar bitwidth is
deterministic from cache storage layout (no attention-based selection).
"""

import json
from pathlib import Path

PHYSICAL_BITS_PER_SCALAR: dict[str, float] = {
    "qserve_kv4": 4.50,
    "turboquant": 4.375,
    "headq": 3.25,
    "cachegen": 6.392,
}


def patch_longmem_file(path: Path, bits_per_scalar: float) -> bool:
    """Patch diagnostic bitwidth/compression fields in a single LongMemEval JSON."""
    if not path.exists():
        print(f"SKIP (missing): {path}")
        return False

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    changed = False
    for entry in data.get("results", []):
        extra = entry.get("extra")
        if not isinstance(extra, dict):
            continue
        retention = float(extra.get("diag_mean_overall_retention_ratio", 1.0))
        new_bitwidth = round(retention * bits_per_scalar, 2)
        new_comp_ratio = round(16.0 / max(0.01, new_bitwidth), 2)

        if extra.get("diag_mean_overall_effective_bitwidth") != new_bitwidth:
            extra["diag_mean_overall_effective_bitwidth"] = new_bitwidth
            changed = True
        if extra.get("diag_mean_min_session_effective_bitwidth") != new_bitwidth:
            extra["diag_mean_min_session_effective_bitwidth"] = new_bitwidth
            changed = True
        if extra.get("diag_mean_compression_ratio") != new_comp_ratio:
            extra["diag_mean_compression_ratio"] = new_comp_ratio
            changed = True
        for legacy_key in ("compression_ratio_vs_bf16", "compression_ratio"):
            if legacy_key in extra:
                del extra[legacy_key]
                changed = True

    if changed:
        with path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
            f.write("\n")
        print(f"PATCHED: {path} (bits_per_scalar={bits_per_scalar})")
    else:
        print(f"OK (already patched): {path}")
    return changed


def main() -> None:
    results_dir = Path("results")
    for tech, bits in PHYSICAL_BITS_PER_SCALAR.items():
        patch_longmem_file(results_dir / f"{tech}_longmem_pytorch_16gb.json", bits)


if __name__ == "__main__":
    main()
