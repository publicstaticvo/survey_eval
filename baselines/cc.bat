@echo off
setlocal EnableExtensions EnableDelayedExpansion

set "SCRIPT_DIR=%~dp0"
set "TEST_INPUTS_DIR=%SCRIPT_DIR%..\agent\test_inputs"

pushd "%SCRIPT_DIR%" >nul

for %%G in (pdf_content codex_surveys sgen_surveys cc_surveys) do (
    set "SURVEYS_DIR=%TEST_INPUTS_DIR%\%%G"
    if not exist "!SURVEYS_DIR!\" (
        echo surveys directory not found: "!SURVEYS_DIR!"
        exit /b 1
    )

    for /d %%S in ("!SURVEYS_DIR!\*") do (
        for %%M in (PLAIN ARISE TRUSTSURVEY) do (
            echo group=%%G input="%%~fS" output-file="%%~nS.json" mode=%%M
            python llmeval.py --input-dir "%%~fS" --output-root "%SCRIPT_DIR%\baselines_3\cc\%%G" --output-file "%%~nS.json" --mode %%M --backend claude-code
        )
    )

    for %%S in ("!SURVEYS_DIR!\*.json") do (
        for %%M in (PLAIN ARISE TRUSTSURVEY) do (
            echo group=%%G input="%%~fS" output-file="%%~nS.json" mode=%%M
            python llmeval.py --input-dir "%%~fS" --output-root "%SCRIPT_DIR%\baselines_3\cc\%%G" --output-file "%%~nS.json" --mode %%M --backend claude-code
        )
    )
)

popd >nul
endlocal
