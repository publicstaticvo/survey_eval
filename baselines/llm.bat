@echo off
setlocal EnableExtensions EnableDelayedExpansion

set "SCRIPT_DIR=%~dp0"
set "CC_SURVEYS_DIR=%SCRIPT_DIR%..\..\cc_surveys"

if not exist "%CC_SURVEYS_DIR%\" (
    echo cc_surveys directory not found: "%CC_SURVEYS_DIR%"
    exit /b 1
)

pushd "%SCRIPT_DIR%" >nul

for /d %%S in ("%CC_SURVEYS_DIR%\*") do (
    for %%M in (TRUSTSURVEY) do (
        echo input-dir="%%~fS" output-file="%%~nS.json" mode=%%M
        python llmeval.py --input-dir "%%~fS" --output-root "%SCRIPT_DIR%\cc\cc" --output-file "%%~nS.json" --mode %%M
    )
)

popd >nul
endlocal
