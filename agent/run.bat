@echo off
setlocal EnableExtensions EnableDelayedExpansion

python main.py --batch-dir test_inputs/codex_surveys --no-parallel --run-modules 2

python main.py --batch-dir test_inputs/sgen_surveys --no-parallel --run-modules 2

python main.py --batch-dir test_inputs/pdf_content --no-parallel --run-modules 2

python main.py --batch-dir test_inputs/cc_surveys --no-parallel --run-modules 2
