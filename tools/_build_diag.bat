@echo off
setlocal
rem Rebuild llama-sabah-diag after a ggml-cuda change, with the same MSVC
rem discovery as build_dll.bat. The build tree is third_party\llama.cpp\build.
rem
rem CMAKE_CUDA_ARCHITECTURES is forced to 120: this machine is sm_120 and the
rem cached default (120;61) compiles every CUDA translation unit twice, once
rem for Pascal, which nothing here can run. Dropping it halves the build.
set VSDIR=D:\Program Files\Microsoft Visual Studio\2022\Professional
if not exist "%VSDIR%\VC\Auxiliary\Build\vcvars64.bat" set VSDIR=D:\Program Files\Microsoft Visual Studio\18\Community
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

set "SRC=%~dp0..\third_party\llama.cpp"
if /i "%~1"=="noreconf" goto build
echo [%TIME%] reconfigure: CMAKE_CUDA_ARCHITECTURES=120
cmake -S "%SRC%" -B "%SRC%\build" -DCMAKE_CUDA_ARCHITECTURES=120 -DGGML_CUDA_GRAPHS=ON > "%TEMP%\sabah-reconf.log" 2>&1
if errorlevel 1 ( echo RECONFIGURE FAILED & type "%TEMP%\sabah-reconf.log" & exit /b 1 )
:build
echo [%TIME%] build llama-sabah-diag
cmake --build "%SRC%\build" --target llama-sabah-diag -- -j 8
echo [%TIME%] BUILD_EXIT=%ERRORLEVEL%
endlocal
exit /b %ERRORLEVEL%
