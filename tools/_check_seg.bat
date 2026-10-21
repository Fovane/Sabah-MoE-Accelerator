@echo off
rem Syntax-only check of the host-only CUDA helper TUs (no linking, no ninja).
setlocal
set VSDIR=D:\Program Files\Microsoft Visual Studio\2022\Professional
if not exist "%VSDIR%\VC\Auxiliary\Build\vcvars64.bat" set VSDIR=D:\Program Files\Microsoft Visual Studio\18\Community
set VCTOOLSET=
if not "%VSDIR:\18\=%"=="%VSDIR%" set VCTOOLSET=-vcvars_ver=14.44
call "%VSDIR%\VC\Auxiliary\Build\vcvars64.bat" %VCTOOLSET% >nul
if errorlevel 1 ( echo VCVARS FAILED & exit /b 1 )
set ROOT=%~dp0..\third_party\llama.cpp
set CUDAROOT=D:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8
cl /nologo /TP /c /std:c++17 /EHsc /MD /utf-8 /W3 /I "%ROOT%\ggml\include" /I "%ROOT%\ggml\src" /I "%ROOT%\ggml\src\ggml-cuda" /I "%CUDAROOT%\include" /Fo"%TEMP%\sabah_seg_check.obj" "%ROOT%\ggml\src\ggml-cuda\sabah-seg.cpp"
echo CL_EXIT=%ERRORLEVEL%
exit /b %ERRORLEVEL%
