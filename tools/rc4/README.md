# RC4 diagnostic harness

The scripts that produced `results/rc4_numerical/`. They drive
`llama-sabah-diag` (from `patches/llama.cpp/0002-sabah-diag-capture-tool.patch`)
and write their runs to `D:/sabah_rc4/runs/`. Paths are constants at the top of
each file; adjust them for another machine.

| script | purpose |
|---|---|
| `run_mode.py` | run one path — `cpu`, `cuda`, `sabah` or `stock` — with identical settings; only placement and executor variables differ |
| `analyze.py` | router, ladder and logit comparison between two runs (id-aligned) |
| `opcheck.py` | replay each captured `MUL_MAT_ID` in float64 on its own inputs; activation-quantization emulations |
| `summarize.py` | write router / block / expert / logit evidence |
| `seq_analyze.py` | greedy equivalence, determinism, teacher-forced error growth |
| `bench_analyze.py` | sustained-generation throughput and measured speedup |
| `api_ab.py` | `sabah serve` reference vs sabah through the OpenAI API |

Typical ladder run (prefill and one decode step, all tensors, `MUL_MAT_ID`
inputs):

```bash
python run_mode.py sabah runs/ladder_sabah -n 2 --steps 0,1 --mmid --verify-bytes \
  --names ffn_moe_topk,ffn_moe_weights_norm,ffn_moe_gate,ffn_moe_up,ffn_moe_down,ffn_moe_out,ffn_out,l_last
```

Token equivalence and timing are always measured without `--names`: observing
a tensor splits the graph there and disables CUDA fusion across it.
