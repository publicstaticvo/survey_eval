@echo off
setlocal EnableExtensions EnableDelayedExpansion

set "SCRIPT_DIR=%~dp0"
set "TEST_INPUTS_DIR=%SCRIPT_DIR%..\agent\test_inputs"

pushd "%SCRIPT_DIR%" >nul

for %%O in (baselines_4 baselines_5) do (
    for %%G in (codex_surveys sgen_surveys cc_surveys pdf_content) do (
        set "SURVEYS_DIR=%TEST_INPUTS_DIR%\%%G"
        if not exist "!SURVEYS_DIR!\" (
            echo surveys directory not found: "!SURVEYS_DIR!"
            exit /b 1
        )

        for /d %%S in ("!SURVEYS_DIR!\*") do (
            for %%M in (PLAIN ARISE TRUSTSURVEY) do (
                echo group=%%G input="%%~fS" output-file="%%~nS.json" mode=%%M
                python llmeval.py --input-dir "%%~fS" --output-root "%SCRIPT_DIR%\%%O\llm\%%G" --output-file "%%~nS.json" --mode %%M --backend llm --max-concurrency 10 --render-paper-markdown
            )
        )

        for %%S in ("!SURVEYS_DIR!\*.json") do (
            for %%M in (PLAIN ARISE TRUSTSURVEY) do (
                echo group=%%G input="%%~fS" output-file="%%~nS.json" mode=%%M
                python llmeval.py --input-dir "%%~fS" --output-root "%SCRIPT_DIR%\%%O\llm\%%G" --output-file "%%~nS.json" --mode %%M --backend llm --max-concurrency 10 --render-paper-markdown
            )
        )

        @REM for /d %%S in ("!SURVEYS_DIR!\*") do (
        @REM     for %%M in (PLAIN ARISE TRUSTSURVEY) do (
        @REM         echo group=%%G input="%%~fS" output-file="%%~nS.json" mode=%%M
        @REM         python llmeval.py --input-dir "%%~fS" --output-root "%SCRIPT_DIR%\%%O\cc\%%G" --output-file "%%~nS.json" --mode %%M --backend claude-code
        @REM     )
        @REM )

        @REM for %%S in ("!SURVEYS_DIR!\*.json") do (
        @REM     for %%M in (PLAIN ARISE TRUSTSURVEY) do (
        @REM         echo group=%%G input="%%~fS" output-file="%%~nS.json" mode=%%M
        @REM         python llmeval.py --input-dir "%%~fS" --output-root "%SCRIPT_DIR%\%%O\cc\%%G" --output-file "%%~nS.json" --mode %%M --backend claude-code
        @REM     )
        @REM )
    )
)



popd >nul
endlocal
