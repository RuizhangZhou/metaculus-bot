@echo off
setlocal enabledelayedexpansion

set "ROOT=C:\Users\zr-admin\source\repos\metaculus\metac-bot-template"
set "LOG_DIR=%ROOT%\logs"
set "LOG_FILE=%LOG_DIR%\market_pulse_daily.log"

if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"

echo.>>"%LOG_FILE%"
echo ================================================================================>>"%LOG_FILE%"
echo [%DATE% %TIME%] Market Pulse daily run: START>>"%LOG_FILE%"

cd /d "%ROOT%"

set "SMART_SEARCHER_NUM_SEARCHES=1"
set "SMART_SEARCHER_NUM_SITES_PER_SEARCH=5"
set "SMART_SEARCHER_USE_ADVANCED_FILTERS=false"

set "PYTHONUTF8=1"

set "BOT_REQUIRE_KICONNECT=true"
set "BOT_ENABLE_FALLBACK=true"
set "BOT_FALLBACK_MODEL=openrouter/openai/gpt-oss-120b:free"

call "%ROOT%\.venv\Scripts\python.exe" "%ROOT%\main.py" --mode tournament_update --researcher "smart-searcher/kiconnect" --submit >>"%LOG_FILE%" 2>&1
set "EXIT_CODE=%ERRORLEVEL%"

echo.>>"%LOG_FILE%"
echo [%DATE% %TIME%] Market Pulse retrospective run: START>>"%LOG_FILE%"
call "%ROOT%\.venv\Scripts\python.exe" "%ROOT%\main.py" --mode retrospective >>"%LOG_FILE%" 2>&1
set "RETRO_EXIT_CODE=%ERRORLEVEL%"
echo [%DATE% %TIME%] Market Pulse retrospective run: END (exit_code=!RETRO_EXIT_CODE!)>>"%LOG_FILE%"

echo [%DATE% %TIME%] Market Pulse daily run: END (exit_code=!EXIT_CODE!)>>"%LOG_FILE%"
exit /b !EXIT_CODE!
