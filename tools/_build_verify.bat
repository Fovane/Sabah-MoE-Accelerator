@echo off
call "$DRIVE\Program Files\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvars64.bat" >nul
cd /d $WORKSPACE\tools
nvcc -O3 -arch=sm_120 -o tile_exact.exe tile_exact.cu
echo VERIFY_BUILD=%ERRORLEVEL%
