@echo off
setlocal
cd /d "%~dp0"

where poetry >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    poetry run python main.py --ui
    goto :done
)

where python >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    python main.py --ui
    goto :done
)

echo Poetry or Python was not found in PATH.
echo Install dependencies first, then try again.
pause

:done
endlocal
