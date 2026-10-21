"""Add the quality caveat, related work, third-party attribution and hygiene notes."""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PATH = os.path.join(ROOT, "README.md")


def rl(old, new, s):
    assert old in s, "NOT FOUND: " + old[:70]
    return s.replace(old, new, 1)


def main():
    s = open(PATH, encoding="utf-8").read()

    # 7. make the quality boundary explicit in the not-claims list
    old = (
        "- **Correctness is about bytes served, not model quality.** Byte-identical means\n"
        "  the runtime serves exactly what the GGUF contains (`verify_ok`,\n"
        "  `id_check`); it is not a perplexity or benchmark comparison, and none is run\n"
        "  here."
    )
    if old not in s:
        # the wording differs slightly; find the bullet by its opening words
        i = s.find("- **Correctness is about bytes served, not model quality.**")
        assert i >= 0, "correctness bullet not found"
        j = s.find("\n- ", i + 10)
        if j < 0:
            j = s.find("\n## ", i + 10)
        old = s[i:j]
    new = (
        "- **Correctness is about bytes served, not model quality.** Byte-identical means the\n"
        "  runtime serves exactly the GGUF's quantised weights (`verify_ok`, `id_check`, and\n"
        "  the `SABAH_PREFILL_GROUP`/SEG toggles all check out byte-equal against the plain\n"
        "  path). It does **not** prove the model is good: no perplexity, no accuracy\n"
        "  benchmark and no A/B against another runtime is run here. The IQ3_S / IQ4_NL\n"
        "  quantisation quality numbers are ISTA-DASLab's own measurements of their\n"
        "  conversion, not ours, and Sabah can only be as good as the file it is handed. Run\n"
        "  a perplexity or task-level comparison against a reference build before drawing\n"
        "  any conclusion about output quality."
    )
    s = s.replace(old, new, 1)

    # 8. related work, clearly labelled as not measured by us
    old = "## Status\n"
    new = (
        "## Related work, and why this is not compared to it\n"
        "\n"
        "| project | what it is | how it differs here |\n"
        "|---|---|---|\n"
        "| [Glyd](https://github.com/antimatter15/glyd) | SSD-resident MoE inference on a\n"
        "  consumer box, flash-attention and RAM-resident experts | different hardware class;\n"
        "  **not measured by us** |\n"
        "| [Strata](https://github.com/bdmendes/strata) | hierarchical SSD -> RAM -> VRAM\n"
        "  caching for this model family | **not measured by us**; Strata's numbers are on\n"
        "  their drives, which are not this drive |\n"
        "| [GyroStrata](https://github.com/antimatter15/gyrostrata) | a fork/continuation of\n"
        "  the Strata approach | **not measured by us** |\n"
        "| [SGLang SSD Expert Pack](https://docs.sglang.ai/advanced_features/ssd_expert_pack.html) |\n"
        "  SSD expert packing inside a serving stack | **not measured by us**; also a different\n"
        "  integration model (server, many concurrent requests) |\n"
        "\n"
        "Nothing in this README is a comparison against any of them. Every number here is\n"
        "this rig, this drive (MSI SPATIUM M371, PCIe gen3 x4, measured 1.05-1.45 GB/s fresh\n"
        "offsets) and this GPU (RTX 5060 Ti 16 GB). A faster drive or a different caching\n"
        "policy would change every figure here.\n"
        "\n"
        "## Status\n"
    )
    s = rl(old, new, s)

    # 9. third-party attribution
    old = (
        "## License\n"
        "\n"
        "See [`LICENSE`](LICENSE).\n"
    )
    new = (
        "## License and third-party components\n"
        "\n"
        "`LICENSE` (MIT) covers the Sabah code in this repository. The build also pieces\n"
        "together third-party components, listed with their licences in\n"
        "[`THIRD_PARTY.md`](THIRD_PARTY.md):\n"
        "\n"
        "- **llama.cpp** - MIT, pinned at `96ffdc4`, with the Sabah patches under\n"
        "  `patches/llama.cpp/` (each patch header records the pinned commit);\n"
        "- **the model** - `ISTA-DASLab/Qwen3.8-Flash-Next` IQ3_S, not bundled; its licence\n"
        "  and the quantisation quality numbers are the quantiser's, not ours;\n"
        "- **CUDA / the NVIDIA driver** - used under their own licences, nothing\n"
        "  redistributed here.\n"
    )
    s = rl(old, new, s)

    open(PATH, "w", encoding="utf-8", newline="\n").write(s)
    print("applied fixes 7, 8, 9")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
