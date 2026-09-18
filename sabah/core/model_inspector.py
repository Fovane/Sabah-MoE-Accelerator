"""
Sabah / core / model_inspector

Reads a GGUF and produces a ModelProfile. Nothing about model geometry is typed
in by the user, and nothing is assumed: an architecture is supported only if it
is explicitly registered, and the physical expert-slicing assumption is
VERIFIED against the tensor table rather than trusted.

Sabah's whole design rests on one physical property: a per-expert slice must be
a contiguous, quantization-block-aligned byte range, so an expert can be fetched
with a few contiguous reads and zero read amplification. If that does not hold
for a model, Sabah must refuse to accelerate it rather than silently apply
Flash-Next assumptions to an arbitrary MoE.
"""
from __future__ import annotations

import os
import re
import sys
import json
import hashlib
import collections
from dataclasses import dataclass, field, asdict
from typing import Optional

PROFILE_VERSION = 1

# elements-per-block, bytes-per-block for the quant types Sabah can reason about
QUANT_BLOCK = {
    "F32": (1, 4), "F16": (1, 2), "BF16": (1, 2),
    "Q4_0": (32, 18), "Q4_1": (32, 20), "Q5_0": (32, 22), "Q5_1": (32, 24),
    "Q8_0": (32, 34), "IQ4_NL": (32, 18),
    "Q2_K": (256, 84), "Q3_K": (256, 110), "Q4_K": (256, 144),
    "Q5_K": (256, 176), "Q6_K": (256, 210), "Q8_K": (256, 292),
}


class UnsupportedModel(Exception):
    """Raised when Sabah cannot safely accelerate a model."""


@dataclass
class ExpertTensorInfo:
    name: str
    block: int
    role: str                 # gate / up / down
    qtype: str
    shape: tuple
    tensor_bytes: int
    per_expert_bytes: int
    shard: int
    offset: int
    contiguous: bool


@dataclass
class ModelProfile:
    version: int = PROFILE_VERSION
    path: str = ""
    shards: list = field(default_factory=list)
    file_bytes: int = 0
    tensor_bytes: int = 0
    n_tensors: int = 0
    architecture: str = ""
    quant_label: str = ""

    n_blocks: int = 0
    d_model: int = 0
    n_experts: int = 0
    n_experts_used: int = 0
    expert_ff: int = 0
    vocab: int = 0
    context_length: int = 0

    # byte budget
    expert_bank_bytes: int = 0
    fixed_bytes: int = 0
    lookup_bytes: int = 0
    per_block_expert_bytes: dict = field(default_factory=dict)
    per_token_expert_bytes: int = 0
    role_bytes: dict = field(default_factory=dict)

    experts_contiguous: bool = False
    expert_tensors: list = field(default_factory=list)
    supported: bool = False
    notes: list = field(default_factory=list)

    def to_json(self) -> str:
        d = asdict(self)
        d["expert_tensors"] = [asdict(t) if not isinstance(t, dict) else t
                               for t in self.expert_tensors]
        return json.dumps(d, indent=1)

    @property
    def n_expert_objects(self) -> int:
        return self.n_blocks * self.n_experts


# --------------------------------------------------------------------------
# architecture registry: only what has been verified is accelerated
# --------------------------------------------------------------------------
@dataclass
class ArchSpec:
    name: str
    kv_prefix: str
    gate_pat: str
    up_pat: str
    down_pat: str
    note: str = ""


ARCHS = {
    "qwen4exp": ArchSpec(
        name="qwen4exp",
        kv_prefix="qwen4exp",
        gate_pat=r"blk\.(\d+)\.ffn_gate_exps\.weight",
        up_pat=r"blk\.(\d+)\.ffn_up_exps\.weight",
        down_pat=r"blk\.(\d+)\.ffn_down_exps\.weight",
        note="Qwen3.8-Flash-Next family; verified in Sabah v3/v4",
    ),
}


def _gguf_reader():
    for cand in (r"D:/llama-glm53/gguf-py", os.environ.get("SABAH_GGUF_PY", "")):
        if cand and os.path.isdir(cand) and cand not in sys.path:
            sys.path.insert(0, cand)
    try:
        from gguf import GGUFReader          # noqa
        return GGUFReader
    except Exception as e:                   # pragma: no cover
        raise UnsupportedModel(
            "gguf-py not importable (%s). Set SABAH_GGUF_PY to a gguf-py "
            "checkout." % e)


def _shard_paths(path: str) -> list:
    """Expand a split GGUF (…-00001-of-000NN.gguf) into all its shards."""
    m = re.match(r"(.*)-(\d{5})-of-(\d{5})\.gguf$", os.path.basename(path))
    if not m:
        return [path]
    base, _, total = m.group(1), m.group(2), int(m.group(3))
    d = os.path.dirname(path)
    out = []
    for i in range(1, total + 1):
        p = os.path.join(d, "%s-%05d-of-%05d.gguf" % (base, i, total))
        if os.path.exists(p):
            out.append(p)
    return out or [path]


def _kv(reader, key, default=None):
    f = reader.fields.get(key)
    if f is None:
        return default
    try:
        return f.contents()
    except Exception:
        return default


def inspect_model(path: str, verbose: bool = False) -> ModelProfile:
    GGUFReader = _gguf_reader()
    if not os.path.exists(path):
        raise UnsupportedModel("model not found: %s" % path)

    shards = _shard_paths(path)
    prof = ModelProfile(path=os.path.abspath(path),
                        shards=[os.path.basename(p) for p in shards])
    prof.file_bytes = sum(os.path.getsize(p) for p in shards)

    tensors = {}
    arch = None
    head = None
    for si, p in enumerate(shards):
        r = GGUFReader(p, "r")
        if head is None:
            head = r
            arch = _kv(r, "general.architecture")
        for t in r.tensors:
            tensors[t.name] = dict(
                shard=si, offset=int(t.data_offset), bytes=int(t.n_bytes),
                nelem=int(t.n_elements), qtype=t.tensor_type.name,
                shape=tuple(int(x) for x in t.shape))
    prof.n_tensors = len(tensors)
    prof.tensor_bytes = sum(t["bytes"] for t in tensors.values())
    prof.architecture = arch or "?"
    prof.quant_label = str(_kv(head, "general.file_type", "?"))

    spec = ARCHS.get(prof.architecture)
    if spec is None:
        prof.supported = False
        prof.notes.append(
            "architecture %r is not in Sabah's verified registry. Supported: %s. "
            "Sabah will not apply another model's expert-layout assumptions."
            % (prof.architecture, ", ".join(sorted(ARCHS))))
        return prof

    kp = spec.kv_prefix
    prof.n_blocks = int(_kv(head, "%s.block_count" % kp, 0) or 0)
    prof.d_model = int(_kv(head, "%s.embedding_length" % kp, 0) or 0)
    prof.n_experts = int(_kv(head, "%s.expert_count" % kp, 0) or 0)
    prof.n_experts_used = int(_kv(head, "%s.expert_used_count" % kp, 0) or 0)
    prof.expert_ff = int(_kv(head, "%s.expert_feed_forward_length" % kp, 0) or 0)
    prof.context_length = int(_kv(head, "%s.context_length" % kp, 0) or 0)
    emb = tensors.get("token_embd.weight")
    if emb:
        prof.vocab = int(emb["shape"][1])

    if not (prof.n_experts and prof.n_experts_used and prof.n_blocks):
        prof.supported = False
        prof.notes.append("model does not declare MoE expert counts; not a "
                          "routed-MoE artifact Sabah can accelerate")
        return prof

    # ---------------- expert tensors, and the contiguity proof -------------
    pats = [(spec.gate_pat, "gate"), (spec.up_pat, "up"), (spec.down_pat, "down")]
    exp_info, blk_cost = [], collections.defaultdict(int)
    all_contig = True
    for name, t in tensors.items():
        for pat, role in pats:
            m = re.fullmatch(pat, name)
            if not m:
                continue
            blk = int(m.group(1))
            shp = t["shape"]
            if len(shp) < 3 or shp[2] != prof.n_experts:
                all_contig = False
                prof.notes.append("%s: expert axis is not the slowest "
                                  "dimension (shape %s)" % (name, shp))
                continue
            ebk = QUANT_BLOCK.get(t["qtype"])
            if ebk is None:
                all_contig = False
                prof.notes.append("%s: unknown quant type %s" % (name, t["qtype"]))
                continue
            elems, blkbytes = shp[0] * shp[1], ebk[1]
            per_expert = elems // ebk[0] * blkbytes
            ok = (elems % ebk[0] == 0 and shp[0] % ebk[0] == 0
                  and per_expert * prof.n_experts == t["bytes"])
            if not ok:
                all_contig = False
                prof.notes.append(
                    "%s: per-expert slice is not quant-block aligned "
                    "(elems=%d blk=%d bytes=%d)" % (name, elems, ebk[0], t["bytes"]))
            exp_info.append(ExpertTensorInfo(
                name=name, block=blk, role=role, qtype=t["qtype"], shape=shp,
                tensor_bytes=t["bytes"], per_expert_bytes=per_expert,
                shard=t["shard"], offset=t["offset"], contiguous=bool(ok)))
            blk_cost[blk] += per_expert

    prof.expert_tensors = [asdict(e) for e in exp_info]
    prof.experts_contiguous = bool(all_contig and exp_info)
    prof.per_block_expert_bytes = {int(k): int(v) for k, v in blk_cost.items()}
    prof.expert_bank_bytes = sum(e.tensor_bytes for e in exp_info)
    prof.per_token_expert_bytes = sum(blk_cost.values()) * prof.n_experts_used

    # ---------------- fixed vs lookup vs expert ---------------------------
    exp_names = {e.name for e in exp_info}
    roles = collections.Counter()
    fixed = lookup = 0
    for name, t in tensors.items():
        if name in exp_names:
            roles["routed_experts"] += t["bytes"]
            continue
        if name.startswith("token_embd") or "per_layer_token_embd" in name:
            roles["lookup"] += t["bytes"]; lookup += t["bytes"]
        else:
            roles["fixed"] += t["bytes"]; fixed += t["bytes"]
    prof.fixed_bytes = fixed
    prof.lookup_bytes = lookup
    prof.role_bytes = dict(roles)

    if not prof.experts_contiguous:
        prof.supported = False
        prof.notes.append(
            "expert slices are not contiguous/aligned; Sabah's streaming design "
            "does not apply to this artifact")
    else:
        prof.supported = True
        prof.notes.append("expert slices verified contiguous and quant-block "
                          "aligned across %d expert tensors" % len(exp_info))
    return prof


def summarize(prof: ModelProfile) -> str:
    L = []
    a = L.append
    a("model      : %s" % os.path.basename(prof.path))
    a("shards     : %d, %.3f GB on disk, %d tensors"
      % (len(prof.shards), prof.file_bytes / 1e9, prof.n_tensors))
    a("arch       : %s   supported: %s"
      % (prof.architecture, "YES" if prof.supported else "NO"))
    if prof.n_experts:
        a("geometry   : %d blocks, d_model %d, %d experts, top-%d, expert_ff %d"
          % (prof.n_blocks, prof.d_model, prof.n_experts,
             prof.n_experts_used, prof.expert_ff))
        costs = sorted(set(prof.per_block_expert_bytes.values()))
        a("expert bank: %.3f GB in %d objects; per-expert %s bytes"
          % (prof.expert_bank_bytes / 1e9, prof.n_expert_objects,
             "/".join("%,d".replace("%,d", "{:,}").format(c) for c in costs)))
        a("per token  : %.4f GB of routed expert weight"
          % (prof.per_token_expert_bytes / 1e9))
        a("fixed path : %.4f GB (read every token)" % (prof.fixed_bytes / 1e9))
        a("lookup     : %.4f GB (row gathers, not streamed)" % (prof.lookup_bytes / 1e9))
        a("contiguous : %s" % ("YES" if prof.experts_contiguous else "NO"))
    for n in prof.notes:
        a("note       : %s" % n)
    return "\n".join(L)


if __name__ == "__main__":
    p = sys.argv[1] if len(sys.argv) > 1 else ""
    prof = inspect_model(p)
    print(summarize(prof))
