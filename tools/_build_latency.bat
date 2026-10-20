@echo off
setlocal
set VSDIR=C:\Program Files\Microsoft Visual Studio\2022\Professional
if not exist "%VSDIR%\VC\Auxiliary\Build\vcvars64.bat" set VSDIR=C:\Program Files\Microsoft Visual Studio\18\Community
if exist "%VSDIR%\VC\Auxiliary\Build\vcvars64.bat" goto have_vs
set "VSDIR="
set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
if exist "%VSWHERE%" for /f "usebackq delims=" %%I in (`"%VSWHERE%" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do set "VSDIR=%%I"
if not defined VSDIR ( echo NO MSVC FOUND & exit /b 1 )
:have_vs
set VCTOOLSET=
if not "%VSDIR:\18\=%"=="%VSDIR%" set VCTOOLSET=-vcvars_ver=14.44
call "%VSDIR%\VC\Auxiliary\Build\vcvars64.bat" %VCTOOLSET% >nul
if errorlevel 1 ( echo VCVARS FAILED & exit /b 1 )
nvcc -O3 -arch=sm_120 -o "%~dp0launch_latency.exe" "%~dp0launch_latency.cu"
echo NVCC_EXIT=%ERRORLEVEL%
endlocal
exit /b %ERRORLEVEL%
