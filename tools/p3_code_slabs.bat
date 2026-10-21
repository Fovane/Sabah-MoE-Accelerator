@echo off
rem Phase 3.3: the coding trace's hit rate and steady decode vs the slab size.
rem
rem item6c-code-2048 measured the coding trace at slab 6.0e9 and never warmed up:
rem 271.69 ms/token over the last 1920 of 2048 steps. This repeats that run with
rem the same configuration (-c 131072, -n 2048, -b 2048, -ub 512, -t 8, TILE3,
rem SEG default) at a larger slab, so the README can state the limitation with a
rem number instead of a word.
rem
rem Run after any other GPU work: two concurrent runs measure each other.
rem   tools\p3_code_slabs.bat

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
if not "%VSDIR:\18=%"=="%VSDIR%" set VCTOOLSET=-vcvars_ver=14.44
call "%VSDIR%\VC\Auxiliary\Build\vcvars64.bat" %VCTOOLSET% >nul
cd /d "%~dp0..\third_party\llama.cpp\build"
ninja llama-sabah-diag >nul
if errorlevel 1 ( echo BUILD FAILED & exit /b 1 )
cd /d "%~dp0.."

rem the reference run, same configuration as item6c-code-2048
python tools\p3_warmup.py --traces code --slabs 6.0e9 --tokens 2048 --context 131072 --batch 2048 --ub 512
rem the larger slab, to see whether the working set simply did not fit
python tools\p3_warmup.py --traces code --slabs 9.8e9 --tokens 2048 --context 131072 --batch 2048 --ub 512

echo done: see the regime table above
