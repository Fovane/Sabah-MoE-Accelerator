@echo off
REM Build the three TEST-ONLY sentinel libraries used by
REM tests/test_mul_mat_id_contract.py to prove the engine rejects a wrong
REM expert / wrong input row. Never ship these; the product build is
REM build_dll.bat, whose sabah_build_info() reports sentinel=0.
REM
REM usage: build_sentinels.bat [outdir] [arch]
setlocal
pushd "%~dp0"
set OUTDIR=%~1
if "%OUTDIR%"=="" set OUTDIR=%~dp0..\..\..\build\sentinels
set ARCH=%~2
if "%ARCH%"=="" set ARCH=sm_120

set VSDIR=C:\Program Files\Microsoft Visual Studio\2022\Professional
if not exist "%VSDIR%\VC\Auxiliary\Build\vcvars64.bat" set VSDIR=C:\Program Files\Microsoft Visual Studio\18\Community
REM CUDA 12.8 rejects MSVC > 14.4x, so pin the toolset when VS18 is used.
if /i "%VSDIR%"=="C:\Program Files\Microsoft Visual Studio\18\Community" set VCTOOLSET=-vcvars_ver=14.44
call "%VSDIR%\VC\Auxiliary\Build\vcvars64.bat" %VCTOOLSET% >nul
if errorlevel 1 ( echo VCVARS FAILED & popd & exit /b 1 )

if not exist "%OUTDIR%" mkdir "%OUTDIR%"
for %%N in (1 2 3) do (
    echo building sentinel %%N -^> "%OUTDIR%\sabah_rt_sentinel%%N.dll"
    nvcc -O3 -arch=%ARCH% -shared -Xcompiler "/MD" -DSABAH_SENTINEL=%%N ^
         -o "%OUTDIR%\sabah_rt_sentinel%%N.dll" sabah_rt.cu
    if errorlevel 1 ( echo SENTINEL %%N FAILED & popd & exit /b 1 )
)
echo SENTINELS_OK %OUTDIR%
popd
endlocal
