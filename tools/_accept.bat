@echo off
cd /d C:\Projects\quantization\Sabah-MoE-Accelerator
set PYTHONUNBUFFERED=1
set SABAH_CUDA_GRAPH_SEG_STATS=1

set SABAH_CUDA_GRAPH_SEG=0
.venv\Scripts\python.exe tools\phase0_measure.py --label accept-c128k-ref --context 131072 --tokens 1024 --prompt-file prompts\chat_multiturn.txt --hot-bytes 6.0e9 --out-root results\phase0 || exit /b 1

set SABAH_CUDA_GRAPH_SEG=48
.venv\Scripts\python.exe tools\phase0_measure.py --label accept-c128k-seg48 --context 131072 --tokens 1024 --prompt-file prompts\chat_multiturn.txt --hot-bytes 6.0e9 --out-root results\phase0 || exit /b 1

set SABAH_CUDA_GRAPH_SEG=0
.venv\Scripts\python.exe tools\phase0_measure.py --label accept-kv640-ref --context 2048 --tokens 640 --prompt "Hello" --hot-bytes 6.0e9 --out-root results\phase0 || exit /b 1

set SABAH_CUDA_GRAPH_SEG=48
.venv\Scripts\python.exe tools\phase0_measure.py --label accept-kv640-seg48 --context 2048 --tokens 640 --prompt "Hello" --hot-bytes 6.0e9 --out-root results\phase0 || exit /b 1

.venv\Scripts\python.exe tools\phase0_measure.py --label accept-chat2048-seg48 --context 131072 --tokens 2048 --prompt-file prompts\chat_multiturn.txt --hot-bytes 6.0e9 --out-root results\phase0 || exit /b 1

echo DONE
