@echo off
setlocal enabledelayedexpansion

for %%I in ("%~dp0..") do set "ROOT=%%~fI"
set "LOG_DIR=%ROOT%\logs"
set "LOG_FILE=%LOG_DIR%\weekly_retrospective.log"

if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"

echo.>>"%LOG_FILE%"
echo ================================================================================>>"%LOG_FILE%"
echo [%DATE% %TIME%] Weekly retrospective run: START>>"%LOG_FILE%"

cd /d "%ROOT%"

set "PYTHONUTF8=1"

call "%ROOT%\.venv\Scripts\python.exe" "%ROOT%\main.py" --mode weekly_retrospective >>"%LOG_FILE%" 2>&1
set "EXIT_CODE=%ERRORLEVEL%"

echo [%DATE% %TIME%] Weekly retrospective run: END (exit_code=!EXIT_CODE!)>>"%LOG_FILE%"
exit /b !EXIT_CODE!

