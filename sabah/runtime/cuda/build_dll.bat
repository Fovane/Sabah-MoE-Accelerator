@echo off
setlocal
set VSDIR=C:\Program Files\Microsoft Visual Studio\2022\Professional
if not exist "%VSDIR%\VC\Auxiliary\Build\vcvars64.bat" set VSDIR=C:\Program Files\Microsoft Visual Studio\18\Community
REM CUDA 12.8 rejects MSVC > 14.4x, so pin the toolset when VS18 is used.
if /i "%VSDIR%"=="C:\Program Files\Microsoft Visual Studio\18\Community" set VCTOOLSET=-vcvars_ver=14.44
call "%VSDIR%\VC\Auxiliary\Build\vcvars64.bat" %VCTOOLSET% >nul
if errorlevel 1 ( echo VCVARS FAILED & exit /b 1 )
echo VSDIR=%VSDIR%
where cl.exe
cl.exe 2>&1 | findstr /C:"Version" 

pushd "%~dp0"
if exist "%~1" set ARCH=%~1
if "%ARCH%"=="" set ARCH=sm_120
echo ARCH=%ARCH%
nvcc -O3 -arch=%ARCH% -shared -Xcompiler "/MD" -o sabah_rt.dll sabah_rt.cu
echo NVCC_EXIT=%ERRORLEVEL%
popd
endlocal
