# How to run it

Short answer: **use the f16 KV cache.** q8 KV is available and documented, but on
this machine it makes decode 6x slower (the reason is measured, not guessed).

## Talking to it: a local server

The working path on this rig is to start `llama-server` directly with the Sabah
environment set - `tools/start_server.ps1` does it for you:

```powershell
powershell tools\start_server.ps1                      # f16 KV, 128K window
powershell tools\start_server.ps1 -Context 32768        # 32K window, faster
powershell tools\start_server.ps1 -Kv q8_0              # opt-in, see REPORT 28
powershell tools\start_server.ps1 -NoGui                # API only, no chat page
```

It leaves you with:

| endpoint | what it is |
|---|---|
| `http://127.0.0.1:8080` | a chat page in the browser (type and go) |
| `http://127.0.0.1:8080/v1/chat/completions` | the OpenAI-compatible API |

Any OpenAI client points at it with `base_url = "http://127.0.0.1:8080/v1"` and
any `model` name: Open WebUI, LM Studio, LibreChat, SillyTavern, `openai-python`,
or curl:

```powershell
curl http://127.0.0.1:8080/v1/chat/completions -H "Content-Type: application/json" -d '{
  "model": "sabah",
  "messages": [{"role":"user","content":"2+2 kacar?"}],
  "max_tokens": 64
}'
```

Stop it with `Ctrl-C` in its window, or
`Get-Process llama-server | Stop-Process`.

### `sabah serve` is not the runnable path here yet

`sabah serve` (the project's own OpenAI proxy, `sabah/server/openai_proxy.py`)
pins the planner's warm blocks into locked host memory before it launches the
backend, and that is more than this driver grants: **9.477 GB wanted** (11 blocks
x 3 roles x 512 experts x 460,800 B) against a measured **5.034 GB** grant, so it
always stops with `cudaHostAlloc: out of memory` (REPORT 29). There is no flag to
shrink or skip it. The direct route above pins nothing in the serving process and
answers, so that is what is documented here.

### What to expect from the first question

The first question after the model loads is the slow one, and it is the fill, not
the decode: the expert tier has to be pulled off the SSD once. Measured on the
server's own `print_timing` lines, not a stopwatch over HTTP (REPORT 30):

| what | context | prompt eval | decode | decode tok/s |
|---|---|---|---|---|
| first question | 131072 | 504.5 ms/tok | 509.3 ms/tok | 1.96 |
| second question | 131072 | 186.5 ms/tok | 467.4 ms/tok | 2.14 |
| 200-token generation | 131072 | 246.0 ms/tok | 509.1 ms/tok | 1.96 |
| second question | 32768 | 149.2 ms/tok | 417.4 ms/tok | 2.40 |

So plan for **about 2 tok/s** on a question. Two things that table says
plainly:

* it does not warm into the 33 tok/s headline - the 200-token run holds 1.96
  tok/s to the last token;
* the context is not the problem: 32K and 128K differ by under 20%.

The GPU is idle while this happens (`4 % util, 487 MHz, P8`), so the wait is the
expert fetch from the SSD, not the card. The 33.3 tok/s headline is a
narrow-prompt measurement (`"Hello"`, 33,557 expert reads); a chat-templated
Turkish prompt routes the wide set (~84,767 reads) and lands here (REPORT 28,
REPORT 30). Expect a rate in this range for real prompts, and use the 32K
window for chatting if you want the smaller of the two.

## One-time setup

```powershell
cd C:\Projects\quantization\Sabah-MoE-Accelerator

install.bat            # runtime DLL + tools, architecture auto-detected
install.bat llama      # plus the pinned llama.cpp and the measurement driver

$env:SABAH_MODEL_DIR = "C:\Strata-data\models\IQ3_S"
```

To make the model path permanent (new PowerShell windows too):

```powershell
[Environment]::SetEnvironmentVariable("SABAH_MODEL_DIR", "C:\Strata-data\models\IQ3_S", "User")
```

## 128K context, f16 KV - the recommended setting

```powershell
$env:SABAH_RT_LIB = "C:\Projects\quantization\Sabah-MoE-Accelerator\sabah\runtime\cuda\sabah_rt.dll"
$env:SABAH_LLAMA_HOT_BYTES = "6.0e9"     # not 9.8e9 at 128K, see the slab table

third_party\llama.cpp\build\bin\llama-sabah-diag.exe -m "C:\Strata-data\models\IQ3_S\Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf" -dev CUDA0 -ngl 99 --cpu-moe -ot "\.per_layer_token_embd\.weight=CPU" -t 8 -c 131072 -b 2048 -n 512 -p "your prompt"
```

Or through the launcher, which splits the three regimes for you:

```powershell
.venv\Scripts\sabah.exe run "C:\Strata-data\models\IQ3_S\Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf" --tokens 512 --context 131072
```

## What to expect (same settings, f16 KV, clocks pinned)

| regime | ms/token | tok/s |
|---|---|---|
| short context (2K), steady | 30.0 | **33.3** |
| 128K window, short prompt, steady | 133.4 | **7.5** |
| filled ~30K KV, tail | 280.5 | 3.6 |

So with a 128K window open, expect roughly 7.5 tok/s. The 33 tok/s figure is the
short-context case only. As the window grows the attention and indexer work
grows, and a long prompt also puts the fill on the disk.

## If you want q8 KV

Add two flags to the same command:

```powershell
-ctk q8_0 -ctv q8_0
```

What it costs (same prompt, same context, measured, REPORT 28):

| KV | ms/token | tok/s | expert reads |
|---|---|---|---|
| f16 | 30.05 | 33.3 | 33,557 |
| q8_0 | 188.32 | 5.3 | 84,055 |

**Why:** q8 quantises the keys and values, which perturbs the attention output,
which perturbs the hidden state, and with it which experts the router selects.
The same expert ask lands on a wider, less cache-resident set, the misses go to
the disk, and the disk is already the wall - so decode gets 6x slower. The VRAM
it frees is small (the whole 128K KV cache is ~384 MB f16, REPORT 23.1).

Use q8 as a numerically-equivalent mode, not as a speed setting.

## Slab size (this matters)

The slab is the VRAM you reserve for experts. At 128K it does not fit beside the
KV cache and the compute buffer, and the rest moves into **system RAM**
(measured, REPORT 23.1):

| context | slab | note |
|---|---|---|
| `-c 512` .. `-c 32768` | `6.0e9` (9.8e9 also fine) | shared memory +110 / +349 MiB, negligible |
| `-c 65536` | ~3.0e9 | the KV cache starts to dominate |
| `-c 131072` | ~2.4e9 | at 6.0e9 it spills **+3517 MiB** into system RAM |

```powershell
$env:SABAH_LLAMA_HOT_BYTES = "2.4e9"    # recommended for 128K
```

Note: no speed difference was measurable between 2.4e9 and 6.0e9 (both ~221 ms
with q8 KV); at this context the slab changes the Sharing, not the speed.

## Long prompts

Put the prompt in a file and give the chunk size with `-b`:

```powershell
$p = "C:\prompts\long-text.txt"
third_party\llama.cpp\build\bin\llama-sabah-diag.exe -m $env:SABAH_MODEL_DIR\Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf -dev CUDA0 -ngl 99 --cpu-moe -ot "\.per_layer_token_embd\.weight=CPU" -t 8 -c 131072 -b 2048 -n 512 --prompt-file $p
```

A 32K-token prompt took 1174 s (measured). A 128K prompt projects to ~90
minutes, so start with 32K.

`-b 2048` is the default and the right choice: the expert pool is streamed once
per chunk. `-ub 1024` (the sub-batch inside a chunk) cuts disk by 42% but the
outputs are not byte-identical - use it only as a numerically-equivalent mode.

## Clock pinning (required for any measurement)

The same command can take 2x longer depending on the machine state (measured:
88.9 ms vs 58 ms, REPORT 9):

```powershell
nvidia-smi -lgc 2800,3090     # needs admin
nvidia-smi -lmc 13000,14001
```

Without admin, run `python -m sabah.tools.bench_protocol`: it checks whether the
machine is clean, the clock is above 2400 MHz, and whether another process holds
the GPU, and refuses to run when the answer makes the number meaningless.

## Verifying it

```powershell
.venv\Scripts\sabah.exe gate "C:\Strata-data\models\IQ3_S\Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf"
```

Expect `gate: PASS` and `verify 2000/0`. That means the bytes served are the
GGUF's bytes - it does not mean the model is good (a perplexity or benchmark
comparison is a separate job).
