@echo off
setlocal
set VSDIR=$DRIVE\Program Files\Microsoft Visual Studio\2022\Professional
if not exist "%VSDIR%\VC\Auxiliary\Build\vcvars64.bat" set VSDIR=$DRIVE\Program Files\Microsoft Visual Studio\18\Community
if exist "%VSDIR%\VC\Auxiliary\Build\vcvars64.bat" goto have_vs
set "VSDIR="
set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
if exist "%VSWHERE%" for /f "usebackq delims=" %%I in (`"%VSWHERE%" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do set "VSDIR=%%I"
:have_vs
set VCTOOLSET=
if not "%VSDIR:\18\=%"=="%VSDIR%" set VCTOOLSET=-vcvars_ver=14.44
call "%VSDIR%\VC\Auxiliary\Build\vcvars64.bat" %VCTOOLSET% >nul
cd /d $WORKSPACE\third_party\llama.cpp\build
set /p FLAGS=<"$WORKSPACE\tools\_cuda_flags_nofwd.txt"
echo START_SYNTAX %TIME%
nvcc %FLAGS% --syntax-only $WORKSPACE\third_party\llama.cpp\ggml\src\ggml-cuda\ggml-cuda.cu
echo SYNTAX_EXIT=%ERRORLEVEL% END %TIME%
endlocal



