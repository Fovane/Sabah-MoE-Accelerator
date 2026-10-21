@echo off
call "$DRIVE\Program Files\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvars64.bat" >nul
set PATH=$DRIVE\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.4\bin;%PATH%
cd /d $WORKSPACE\tools
nvcc -O3 -arch=sm_120 -o nsys_probe.exe nsys_probe.cu
echo BUILD=%ERRORLEVEL%
