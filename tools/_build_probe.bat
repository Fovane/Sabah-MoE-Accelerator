@echo off
call "D:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvars64.bat" >nul
set PATH=D:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.4\bin;%PATH%
cd /d C:\\Projects\\quantization\\Sabah-MoE-Accelerator\tools
nvcc -O3 -arch=sm_120 -o nsys_probe.exe nsys_probe.cu
echo BUILD=%ERRORLEVEL%
