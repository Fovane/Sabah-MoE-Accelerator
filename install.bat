@echo off
rem ===========================================================================
rem  Sabah Accelerator - one-click installer for Windows
rem
rem  Usage:  install.bat [arch] [llama] [nopause]
rem
rem    arch      nvcc architecture for the runtime DLL, e.g. "sm_120".
rem              Default: the highest compute capability of the installed GPUs.
rem    llama     also clone, patch and build llama.cpp and llama-sabah-diag
rem              (needs git + cmake + ninja; takes 10-40 minutes, a few GB).
rem    nopause   do not wait for a keypress at the end (scripted use).
rem
rem  What it writes, all of it inside this directory:
rem
rem    1. .venv\                                   virtualenv + `pip install -e .`
rem    2. sabah\runtime\cuda\sabah_rt.dll          the streaming runtime
rem    3. sabah\hardware\mgpu_qualify.exe          optional H2D probe
rem    4. third_party\llama.cpp                    only with "llama": the pinned
rem                                                upstream base + patches
rem
rem  Nothing is installed system-wide, nothing touches the registry, and no
rem  model file is ever written: the GGUF is opened read-only at run time.
rem ===========================================================================
setlocal EnableDelayedExpansion
set "SB_ROOT=%~dp0"
if "%SB_ROOT:~-1%"=="\" set "SB_ROOT=%SB_ROOT:~0,-1%"
pushd "%SB_ROOT%" >nul 2>nul
if errorlevel 1 (
    echo install: cannot enter "%SB_ROOT%"
    exit /b 1
)

set "SB_ARCH="
set "SB_ARCH_LIST="
set "SB_WITH_LLAMA=0"
set "SB_DOPAUSE=1"
set "SB_STEP=0"
set "SB_STEPNAME="
set "VSDIR="
set "VCTOOLSET="
set "SB_PYCMD="
set "SB_RC=0"

rem ------------------------------------------------------------------ arguments
:args
if "%~1"=="" goto args_done
set "SB_A=%~1"
if /i "!SB_A!"=="llama"   (shift & set "SB_WITH_LLAMA=1" & goto args)
if /i "!SB_A!"=="nopause" (shift & set "SB_DOPAUSE=0" & goto args)
if /i "!SB_A!"=="help"    goto usage
if /i "!SB_A!"=="-h"      goto usage
if /i "!SB_A!"=="--help"  goto usage
if /i "!SB_A!"=="/?"      goto usage
if /i "!SB_A:~0,3!"=="sm_" (set "SB_ARCH=!SB_A!"    & shift & goto args)
if /i "!SB_A:~0,2!"=="sm"  (set "SB_ARCH=sm_!SB_A:~2!" & shift & goto args)
echo install: unknown argument "!SB_A!"
goto usage

:args_done
call :banner

rem ------------------------------------------------------------------- 1 python
call :step "Python interpreter"
call :find_python
if not defined SB_PYCMD goto fail_python
echo   using: !SB_PYCMD!
%SB_PYCMD% -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)"
if errorlevel 1 (
    echo   Python 3.10 or newer is required, this is:
    %SB_PYCMD% --version 2>&1
    goto fail_python
)

rem -------------------------------------------------------------------- 2 venv
call :step "Virtual environment and the sabah package"
if exist ".venv\Scripts\python.exe" (
    echo   .venv exists, reusing it
) else (
    echo   creating .venv
    %SB_PYCMD% -m venv .venv
    if errorlevel 1 goto fail
)
set "SB_PY=%SB_ROOT%\.venv\Scripts\python.exe"
"%SB_PY%" -m pip install --upgrade pip --disable-pip-version-check --quiet
if errorlevel 1 goto fail
"%SB_PY%" -m pip install -e . --disable-pip-version-check
if errorlevel 1 goto fail
echo   installing gguf-py - 'sabah inspect' and 'sabah plan' read the GGUF header
"%SB_PY%" -m pip install gguf --disable-pip-version-check --quiet
if errorlevel 1 echo   gguf-py not installed - set SABAH_GGUF_PY to a gguf-py checkout instead

rem --------------------------------------------------------------- 3 toolchain
call :step "Compiler toolchain (MSVC + nvcc)"
call :find_vs
if not defined VSDIR goto fail_msvc
echo   visual studio: !VSDIR!
set "VCTOOLSET="
if not "%VSDIR:\18\=%"=="%VSDIR%" set "VCTOOLSET=-vcvars_ver=14.44"
if defined VCTOOLSET echo   pinning the toolset for the CUDA 12.8 front end: !VCTOOLSET!
call "!VSDIR!\VC\Auxiliary\Build\vcvars64.bat" !VCTOOLSET! >nul 2>&1
if errorlevel 1 goto fail_msvc
where cl.exe >nul 2>nul
if errorlevel 1 goto fail_msvc
call :find_nvcc
if errorlevel 1 goto fail_nvcc
for /f "delims=" %%V in ('nvcc --version 2^>^&1 ^| findstr /r /c:"release [0-9]"') do echo  %%V

rem ------------------------------------------------------------------ 4 gpu arch
call :step "GPU architecture"
if defined SB_ARCH (
    echo   from the command line: !SB_ARCH!
    set "SB_ARCH_LIST=!SB_ARCH:sm_=!"
    set "SB_ARCH_LIST=!SB_ARCH_LIST:sm=!"
) else (
    call :detect_arch
    if not defined SB_ARCH goto fail_gpu
    echo   detected: !SB_ARCH!
)
echo   CUDA architectures for a multi-GPU build: !SB_ARCH_LIST!

rem -------------------------------------------------------------- 5 runtime dll
call :step "Streaming runtime (sabah_rt.dll, !SB_ARCH!)"
pushd sabah\runtime\cuda
if not exist sabah_rt.cu (
    popd
    echo   sabah\runtime\cuda\sabah_rt.cu is missing - is this the repository root?
    goto fail
)
if exist sabah_rt.dll move /y sabah_rt.dll sabah_rt.dll.prev >nul 2>nul
call build_dll.bat !SB_ARCH!
if not exist sabah_rt.dll (
    if exist sabah_rt.dll.prev move /y sabah_rt.dll.prev sabah_rt.dll >nul 2>nul
    popd
    echo   the runtime did not build. If nvcc rejected the architecture, retry with
    echo   one your toolkit supports, for example:  install.bat sm_89
    goto fail
)
if exist sabah_rt.dll.prev del /q sabah_rt.dll.prev >nul 2>nul
popd
call :show_artifact "sabah\runtime\cuda\sabah_rt.dll"

rem --------------------------------------------------------------- 6 h2d probe
call :step "H2D probe (mgpu_qualify.exe, optional)"
pushd sabah\hardware
if not exist mgpu_qualify.cu goto h2d_done
nvcc -O3 -arch=!SB_ARCH! -o mgpu_qualify.exe mgpu_qualify.cu >nul 2>&1
if errorlevel 1 goto h2d_failed
call :show_artifact "mgpu_qualify.exe"
goto h2d_done
:h2d_failed
echo   not built - optional; only 'sabah selftest' and 'sabah doctor' use it
:h2d_done
popd

rem ------------------------------------------------------------------ 7 llama
if "!SB_WITH_LLAMA!"=="0" goto after_llama
call :step "llama.cpp and the Sabah bridge"
call :build_llama
if errorlevel 1 goto fail
:after_llama

rem ----------------------------------------------------------------- 8 doctor
call :step "sabah doctor"
"%SB_PY%" -m sabah.tools.cli doctor
echo.
call :summary
goto finish

rem ===========================================================================
rem  subroutines
rem ===========================================================================

:banner
echo.
echo ===========================================================================
echo  Sabah Accelerator - one-click installer
echo ===========================================================================
echo  directory: %SB_ROOT%
if defined SB_ARCH (
    echo  arch:      !SB_ARCH!  ^(from the command line^)
) else (
    echo  arch:      auto-detected from the installed GPUs
)
if "!SB_WITH_LLAMA!"=="1" (
    echo  llama.cpp: also built, pinned upstream base plus both patches
) else (
    echo  llama.cpp: not built - pass "llama" to add it
)
echo ===========================================================================
goto :eof

:step
set /a "SB_STEP+=1"
set "SB_STEPNAME=%~1"
echo.
echo [step !SB_STEP!] %~1
echo ---------------------------------------------------------------------------
goto :eof

rem --- toolchain discovery ---------------------------------------------------

:find_python
set "SB_PYCMD="
where py >nul 2>nul
if not errorlevel 1 set "SB_PYCMD=py -3"
if defined SB_PYCMD goto :eof
where python >nul 2>nul
if not errorlevel 1 set "SB_PYCMD=python"
goto :eof

:find_vs
set "VSDIR="
for %%D in (
    "C:\Program Files\Microsoft Visual Studio\2022\Professional"
    "C:\Program Files\Microsoft Visual Studio\2022\Enterprise"
    "C:\Program Files\Microsoft Visual Studio\2022\Community"
    "C:\Program Files\Microsoft Visual Studio\2022\BuildTools"
    "C:\Program Files\Microsoft Visual Studio\18\Professional"
    "C:\Program Files\Microsoft Visual Studio\18\Enterprise"
    "C:\Program Files\Microsoft Visual Studio\18\Community"
    "C:\Program Files\Microsoft Visual Studio\18\BuildTools"
) do (
    if not defined VSDIR if exist "%%~D\VC\Auxiliary\Build\vcvars64.bat" set "VSDIR=%%~D"
)
if defined VSDIR goto :eof
set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
if not exist "%VSWHERE%" goto :eof
for /f "usebackq delims=" %%I in (`"%VSWHERE%" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do set "VSDIR=%%I"
goto :eof

:find_nvcc
where nvcc >nul 2>nul
if not errorlevel 1 goto :eof
if defined CUDA_PATH if exist "!CUDA_PATH!\bin\nvcc.exe" (
    set "PATH=!CUDA_PATH!\bin;!PATH!"
    goto :eof
)
set "SB_CUDAROOT=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA"
if not exist "!SB_CUDAROOT!" exit /b 1
for /f "delims=" %%D in ('dir /b /ad /o-n "!SB_CUDAROOT!\v*" 2^>nul') do (
    if exist "!SB_CUDAROOT!\%%D\bin\nvcc.exe" (
        echo   nvcc from !SB_CUDAROOT!\%%D
        set "PATH=!SB_CUDAROOT!\%%D\bin;!PATH!"
        exit /b 0
    )
)
exit /b 1

rem --- gpu architecture ------------------------------------------------------

:detect_arch
set "SB_ARCH="
set "SB_ARCH_LIST="
set "SB_BEST=0"
set "SB_CCTMP=%TEMP%\sabah_compute_cap.txt"
if exist "!SB_CCTMP!" del /q "!SB_CCTMP!" >nul 2>nul
nvidia-smi --query-gpu=compute_cap --format=csv,noheader >"!SB_CCTMP!" 2>nul
if not exist "!SB_CCTMP!" goto detect_arch_done
for /f "usebackq tokens=1,2 delims=. " %%A in ("!SB_CCTMP!") do call :cc_try %%A %%B
del /q "!SB_CCTMP!" >nul 2>nul
:detect_arch_done
goto :eof

:cc_try
set "SB_MAJ=%~1"
set "SB_MIN=%~2"
if not defined SB_MIN set "SB_MIN=0"
call :add_arch %SB_MAJ%%SB_MIN%
set /a "SB_N=SB_MAJ*10+SB_MIN" 2>nul
if errorlevel 1 goto :eof
if %SB_N% LEQ %SB_BEST% goto :eof
set "SB_BEST=%SB_N%"
set "SB_ARCH=sm_%SB_MAJ%%SB_MIN%"
goto :eof

:add_arch
if "%~1"=="" goto :eof
if "%SB_ARCH_LIST%"=="" (
    set "SB_ARCH_LIST=%~1"
    goto :eof
)
echo %SB_ARCH_LIST%; | findstr /c:"%~1;" >nul
if errorlevel 1 set "SB_ARCH_LIST=%SB_ARCH_LIST%;%~1"
goto :eof

rem --- the build steps -------------------------------------------------------

:build_llama
set "SB_LROOT=%SB_ROOT%\third_party\llama.cpp"
set "SB_LV=%SB_ROOT%\third_party\llama.cpp.version"
set "SB_LURL="
set "SB_LCOMMIT="
if not exist "!SB_LV!" (
    echo   third_party\llama.cpp.version is missing, cannot pin the upstream base
    exit /b 1
)
for /f "usebackq tokens=1,* delims==" %%K in ("!SB_LV!") do (
    if /i "%%K"=="upstream_url" set "SB_LURL=%%L"
    if /i "%%K"=="base_commit"  set "SB_LCOMMIT=%%L"
)
if not defined SB_LURL (
    echo   no upstream_url in third_party\llama.cpp.version
    exit /b 1
)
if not defined SB_LCOMMIT (
    echo   no base_commit in third_party\llama.cpp.version
    exit /b 1
)
where git >nul 2>nul
if errorlevel 1 (
    echo   git is required for the "llama" option - install it and rerun
    exit /b 1
)
where cmake >nul 2>nul
if errorlevel 1 (
    echo   cmake is required for the "llama" option - install it and rerun
    exit /b 1
)
where ninja >nul 2>nul
if errorlevel 1 (
    echo   ninja is required for the "llama" option - install it and rerun
    exit /b 1
)
echo   upstream: !SB_LURL!
echo   pinned:   !SB_LCOMMIT!

if exist "!SB_LROOT!\.git" goto llama_have_tree
echo   fetching the pinned commit only
if not exist "!SB_LROOT!" mkdir "!SB_LROOT!"
pushd "!SB_LROOT!"
git init -q
if errorlevel 1 ( popd & exit /b 1 )
git remote add origin "!SB_LURL!" >nul 2>nul
git remote set-url origin "!SB_LURL!"
git fetch --depth 1 origin "!SB_LCOMMIT!" >nul 2>nul
if errorlevel 1 (
    echo   shallow fetch of the pinned commit failed, fetching the full history
    git fetch origin
    if errorlevel 1 ( popd & exit /b 1 )
)
git checkout -q "!SB_LCOMMIT!"
if errorlevel 1 (
    echo   cannot check out !SB_LCOMMIT!
    popd
    exit /b 1
)
popd
goto llama_build

:llama_have_tree
echo   reusing the existing checkout at third_party\llama.cpp

:llama_build
pushd "!SB_LROOT!"
call :gitapply "%SB_ROOT%\patches\llama.cpp\0001-sabah-native-mul-mat-id.patch"
if errorlevel 1 ( popd & exit /b 1 )
call :gitapply "%SB_ROOT%\patches\llama.cpp\0002-sabah-diag-capture-tool.patch"
if errorlevel 1 ( popd & exit /b 1 )
echo   configuring
cmake -S . -B build -G Ninja -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=OFF "-DCMAKE_CUDA_ARCHITECTURES=!SB_ARCH_LIST!"
if errorlevel 1 (
    echo   cmake configure failed
    popd
    exit /b 1
)
echo   building - this is the long part
set "SB_JOBS=%SABAH_JOBS%"
if not defined SB_JOBS set "SB_JOBS=%NUMBER_OF_PROCESSORS%"
if not defined SB_JOBS set "SB_JOBS=8"
if %SB_JOBS% GTR 8 set "SB_JOBS=8"
echo   parallel jobs: !SB_JOBS!  (set SABAH_JOBS to change it)
cmake --build build -j !SB_JOBS!
if errorlevel 1 (
    echo   the build failed
    popd
    exit /b 1
)
popd
if not exist "!SB_LROOT!\build\bin\llama-sabah-diag.exe" (
    echo   llama-sabah-diag.exe was not produced
    exit /b 1
)
call :show_artifact "third_party\llama.cpp\build\bin\llama-sabah-diag.exe"
exit /b 0

:gitapply
if not exist "%~1" (
    echo   patch not found: %~1
    exit /b 1
)
git apply --reverse --check "%~1" >nul 2>&1
if not errorlevel 1 (
    echo   already applied: %~nx1
    exit /b 0
)
git apply --check "%~1" >nul 2>&1
if errorlevel 1 (
    echo   %~nx1 does not apply to this tree - refusing to guess
    exit /b 1
)
git apply "%~1"
if errorlevel 1 exit /b 1
echo   applied: %~nx1
exit /b 0

:show_artifact
if not exist "%~1" exit /b 0
for %%F in ("%~1") do echo   %%~nxF  %%~zF bytes
goto :eof

rem --- epilogue --------------------------------------------------------------

:summary
echo ===========================================================================
echo  Done
echo ===========================================================================
echo   .venv\                                     the package and the `sabah` command
echo   sabah\runtime\cuda\sabah_rt.dll            the streaming runtime (!SB_ARCH!)
if "!SB_WITH_LLAMA!"=="1" echo   third_party\llama.cpp\build\bin\llama-sabah-diag.exe   the measurement driver
echo.
echo  To run a model, open a shell here and set the run environment:
echo.
echo     cd /d "%SB_ROOT%"
echo     .venv\Scripts\activate
echo     set SABAH_LLAMA=1
echo     set SABAH_RT_LIB=%SB_ROOT%\sabah\runtime\cuda\sabah_rt.dll
echo     set SABAH_LLAMA_HOT_BYTES=9.8e9
echo     set GGML_OP_OFFLOAD_MIN_BATCH=1
echo     set SABAH_LLAMA_TRACE=1
echo.
echo  Then drive it with the harness, which reports fill, warm-up and steady
echo  state separately instead of one averaged number:
echo.
echo     python -m sabah.tools.cli run    ^<model.gguf^> --tokens 256
echo     python -m sabah.tools.cli gate   ^<model.gguf^>
echo.
echo  SABAH_LLAMA_HOT_BYTES is the expert slab that stays in VRAM; 9.8e9 is the
echo  reference rig's 16 GB card. Size it to your own VRAM - unset it for the 2 GiB
echo  default. Two placement flags matter and are easy to get wrong: GGML_OP_OFFLOAD_MIN_BATCH=1
echo  and, on llama.cpp, -ot "\.per_layer_token_embd\.weight=CPU". Without them
echo  nothing reaches the runtime. See the README, section "Run".
goto :eof

:usage
echo.
echo Usage:  install.bat [arch] [llama] [nopause]
echo.
echo   arch      nvcc architecture for the runtime DLL, e.g. sm_120.
echo             Default: the highest compute capability of the installed GPUs.
echo   llama     also clone, patch and build llama.cpp and llama-sabah-diag.
echo             Needs git + cmake + ninja; takes 10-40 minutes.
echo   nopause   do not wait for a keypress at the end.
echo.
echo   install.bat              the package and the runtime DLL, arch detected
echo   install.bat llama        everything, including the llama.cpp bridge
echo   install.bat sm_89        force an architecture
echo.
set "SB_RC=2"
goto finish

:fail_python
echo   Python was not found on PATH. Install 3.10 or newer from
echo   https://www.python.org/downloads/ and tick "Add python.exe to PATH".
goto fail

:fail_msvc
echo   MSVC was not found. Install "Desktop development with C++" - the free
echo   Build Tools are enough - from https://visualstudio.microsoft.com/downloads/
goto fail

:fail_nvcc
echo   nvcc was not found. Install the CUDA Toolkit 12.8 or newer:
echo   https://developer.nvidia.com/cuda-downloads
goto fail

:fail_gpu
echo   no GPU compute capability was reported by nvidia-smi, so there is no
echo   architecture to build for. Sabah needs a CUDA GPU.
goto fail

:fail
echo.
echo ===========================================================================
echo  INSTALL FAILED at step !SB_STEP! - !SB_STEPNAME!
echo ===========================================================================
set "SB_RC=1"
goto finish

:finish
popd >nul 2>nul
if "!SB_DOPAUSE!"=="1" pause
endlocal & exit /b %SB_RC%
