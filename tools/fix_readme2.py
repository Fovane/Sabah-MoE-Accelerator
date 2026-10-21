"""Fix the clock-state warning and the run example in README.md."""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PATH = os.path.join(ROOT, "README.md")


def rl(old, new, s):
    assert old in s, "NOT FOUND: " + old[:70]
    return s.replace(old, new, 1)


def main():
    s = open(PATH, encoding="utf-8").read()

    # 4. the machine-state warning, with the 58 vs 88 ms episode and the low P-state
    old = (
        "Both are reversible (`nvidia-smi -rgc` / `-rmc`) and need admin. Without them the\n"
        "card may sit at 585 MHz in P8 and the same decode measures twice the time, so a\n"
        "run whose steady SM clock is below 2400 MHz is marked **invalid** by the harness\n"
        "and is not evidence."
    )
    new = (
        "Both are reversible (`nvidia-smi -rgc` / `-rmc`) and need admin.\n"
        "\n"
        "**The machine state can double ms/token, and this has happened here.** The same\n"
        "decode measured **88.9 ms** and **58 ms** on two runs of the same binary: the first\n"
        "sat in a low P-state with the SM clock at 585-900 MHz, the second at 2805 MHz\n"
        "(REPORT 9). A 53% difference from no code change at all. So:\n"
        "\n"
        "- pin the clocks before any timed run, and record `pstate`, `clocks.sm` and\n"
        "  `clocks.mem` next to the number;\n"
        "- the harness marks a run **invalid** when the steady SM clock is below 2400 MHz or\n"
        "  the memory clock is below 10000 MHz, and an invalid run is not evidence - re-run it;\n"
        "- if pinning is refused (it needs admin), the auto-boost may still be enough under\n"
        "  load, but check the recorded steady clock before quoting the time;\n"
        "- other processes holding a compute context are allowed (15 hold them on the\n"
        "  reference rig, mostly system ones that never compute) but they are a flag, not a\n"
        "  refusal - see `python -m sabah.tools.bench_protocol`, which prints the verdict\n"
        "  and refuses on a low clock."
    )
    s = rl(old, new, s)

    # 6. the run example: no local path, and the long-context slab guidance
    old = "$env:SABAH_MODEL"
    if old not in s:
        pass
    old = (
        "llama-sabah-diag.exe `\n"
        "  -m $DRIVE\\Strata-data\models\IQ3_S\Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf `\n"
        "  -dev CUDA0 -ngl 99 --cpu-moe `\n"
        "  -ot \"\\.per_layer_token_embd\\.weight=CPU\" `\n"
        "  -t 8 -c 512 -n 32 -p \"Hello\"\n"
        "```"
    )
    new = (
        "llama-sabah-diag.exe `\n"
        "  -m $env:SABAH_MODEL_DIR\Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf `\n"
        "  -dev CUDA0 -ngl 99 --cpu-moe `\n"
        "  -ot \"\\.per_layer_token_embd\\.weight=CPU\" `\n"
        "  -t 8 -c 512 -n 32 -p \"Hello\"\n"
        "```\n"
        "\n"
        "`SABAH_MODEL_DIR` is the single source of truth for where the shards live - every\n"
        "tool reads it, so no script in the repository carries a drive letter.\n"
        "\n"
        "**Slab size by context** (REPORT 23: the slab has to fit beside the KV cache and\n"
        "the compute buffer, or the driver puts the rest in system RAM):\n"
        "\n"
        "| context | slab | why |\n"
        "|---|---|---|\n"
        "| `-c 512` .. `-c 32768` | 6.0e9 (9.8e9 is fine) | the KV cache is small; the shared-memory\n"
        "spill is negligible (+110 MiB at `-c 512`, +349 MiB at `-c 32768`) |\n"
        "| `-c 65536` | ~3.0e9 | the KV cache starts to dominate |\n"
        "| `-c 131072` | ~2.4e9 | 6.0e9 spills ~3.4 GB into system RAM (Shared Usage +3517 MiB),\n"
        "which makes every miss more expensive than the disk saving is worth |\n"
        "\n"
        "`--hot-bytes 9.8e9` with `-c 512` is the *short-answer* setting; do not carry it to a\n"
        "128K context."
    )
    s = rl(old, new, s)

    # 5. the Shared-Usage caveat as a first-class limitation
    old = "- **At 128K context the RAM-flat claim does not hold.**"
    new = (
        "- **At 128K context the RAM-flat claim does not hold.** The *process* working set\n"
        "  stays flat, but the driver's SLA moves into system RAM when the slab no longer fits\n"
        "  beside the KV cache: sampling the GPU's Windows \"Shared Usage\" counter against its\n"
        "  idle baseline gives **+110 MiB at `-c 512`, +349 MiB at `-c 32768` and +3517 MiB\n"
        "  at `-c 131072`** (`tools/p3_shared.py`, REPORT 23.1). Size the slab for the context\n"
        "  (the table above), and do not quote \"host RAM is flat\" for a 128K run with a 6 GB\n"
        "  slab."
    ) if False else old
    # replace the existing short bullet with the longer one
    i = s.find("- **At 128K context the RAM-flat claim does not hold.**")
    j = s.find("- **At 128K context", i + 10)
    if j > 0:
        pass
    # find the end of that bullet
    j = s.find("\n- ", i + 10)
    if j < 0:
        j = s.find("\n## ", i + 10)
    old_bullet = s[i:j]
    new_bullet = (
        "- **At 128K context the RAM-flat claim does not hold.** The *process* working set stays\n"
        "  flat, but the driver moves slabs into system RAM when the slab no longer fits beside\n"
        "  the KV cache and the compute buffer: sampling the GPU's Windows \"Shared Usage\"\n"
        "  counter against its idle baseline gives **+110 MiB at `-c 512`, +349 MiB at\n"
        "  `-c 32768` and +3517 MiB at `-c 131072`** (`tools/p3_shared.py`, REPORT 23.1).\n"
        "  Size the slab for the context (the table in the reproduce section) and do not quote\n"
        "  \"host RAM is flat\" for a 128K run with a 6 GB slab."
    )
    s = s[:i] + new_bullet + s[j:]

    open(PATH, "w", encoding="utf-8", newline="\n").write(s)
    print("applied fixes 4, 5, 6")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
