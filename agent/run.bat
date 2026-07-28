@echo off
setlocal EnableExtensions EnableDelayedExpansion

python main.py --batch-dir test_inputs/codex_surveys --no-parallel --run-modules 3 --force

python main.py --batch-dir test_inputs/corrupted_surveys --no-parallel --run-modules 0,1,2,3,4

python main.py --batch-dir test_inputs/sgen_surveys --no-parallel --run-modules 0,1,2,3,4

python main.py --batch-dir test_inputs/cc_surveys --no-parallel --run-modules 0,1,2,3,4

