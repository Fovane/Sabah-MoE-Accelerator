"""Compare Sabah and reference router traces without normalizing away errors."""

from __future__ import annotations

import array
import json
import math
import struct
import sys
from pathlib import Path


def floats(path: Path) -> list[float]:
    data = path.read_bytes()
    return list(struct.unpack(f"<{len(data) // 4}f", data))


def ints(path: Path) -> list[int]:
    data = path.read_bytes()
    return list(struct.unpack(f"<{len(data) // 4}i", data))


def main() -> int:
    ref = Path(sys.argv[1])
    sabah = Path(sys.argv[2])
    result: dict[str, object] = {"files": 0, "id_mismatch": 0, "weight_max_abs": 0.0}
    weight_values = 0
    weight_sum_sq = 0.0
    id_first: dict[str, object] | None = None
    id_rows: dict[int, int] = {}
    id_layers: dict[str, dict[str, int]] = {}
    for left in sorted(ref.glob("ffn_moe_topk_L*.i32")):
        right = sabah / left.name
        a, b = ints(left), ints(right)
        if len(a) != len(b):
            raise RuntimeError(f"size mismatch: {left.name}: {len(a)} != {len(b)}")
        layer_count = 0
        layer_first = None
        result["files"] = int(result["files"]) + 1
        for i, (x, y) in enumerate(zip(a, b)):
            if x != y:
                result["id_mismatch"] = int(result["id_mismatch"]) + 1
                layer_count += 1
                if layer_first is None:
                    layer_first = i // 10
                id_rows[i // 10] = id_rows.get(i // 10, 0) + 1
                if id_first is None:
                    id_first = {"file": left.name, "index": i, "reference": x, "sabah": y}
        id_layers[left.name] = {"mismatches": layer_count, "first_row": layer_first}
    result["first_id_mismatch"] = id_first
    result["id_mismatch_rows"] = id_rows
    result["id_layers"] = id_layers
    weight_first: dict[str, object] | None = None
    for left in sorted(ref.glob("ffn_moe_weights_L*.f32")):
        right = sabah / left.name
        a, b = floats(left), floats(right)
        if len(a) != len(b):
            raise RuntimeError(f"size mismatch: {left.name}: {len(a)} != {len(b)}")
        for i, (x, y) in enumerate(zip(a, b)):
            d = abs(x - y)
            if d > float(result["weight_max_abs"]):
                result["weight_max_abs"] = d
            weight_sum_sq += float(d) * float(d)
            weight_values += 1
            if d != 0.0 and weight_first is None:
                weight_first = {"file": left.name, "index": i, "reference": x, "sabah": y, "abs": d}
    result["weight_values"] = weight_values
    result["weight_l2"] = math.sqrt(weight_sum_sq)
    result["first_weight_mismatch"] = weight_first
    for prefix in ("ffn_moe_distill_in", "ffn_moe_logits", "ffn_moe_probs", "ffn_moe_gate", "ffn_moe_up", "ffn_moe_down", "ffn_moe_out"):
        max_abs = 0.0
        l2_sq = 0.0
        count = 0
        first = None
        layer_firsts: dict[str, dict[str, object]] = {}
        for left in sorted(ref.glob(f"{prefix}_L*.f32")):
            right = sabah / left.name
            a, b = floats(left), floats(right)
            if len(a) != len(b):
                raise RuntimeError(f"size mismatch: {left.name}: {len(a)} != {len(b)}")
            for i, (x, y) in enumerate(zip(a, b)):
                d = abs(x - y)
                max_abs = max(max_abs, d)
                l2_sq += d * d
                count += 1
                if d != 0.0 and first is None:
                    first = {"file": left.name, "index": i, "reference": x, "sabah": y, "abs": d}
            local = next((
                {"index": i, "reference": x, "sabah": y, "abs": abs(x - y)}
                for i, (x, y) in enumerate(zip(a, b)) if x != y
            ), None)
            layer_firsts[left.name] = local
        result[prefix] = {"values": count, "max_abs": max_abs, "l2": math.sqrt(l2_sq), "first_mismatch": first, "layer_firsts": layer_firsts}
    ref_manifest = [json.loads(line) for line in (ref / "manifest.jsonl").read_text().splitlines()]
    sabah_manifest = [json.loads(line) for line in (sabah / "manifest.jsonl").read_text().splitlines()]
    result["manifest_equal"] = ref_manifest == sabah_manifest
    result["reference_generated"] = [item["generated"] for item in ref_manifest]
    result["sabah_generated"] = [item["generated"] for item in sabah_manifest]
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["id_mismatch"] == 0 and result["weight_max_abs"] == 0.0 and result["manifest_equal"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
