@echo off
title Q-FUSION Experiment A (V2) Runner
echo ======================================================================
echo Starting Q-FUSION Experiment A (V2) 30-Epoch Benchmark via CMD...
echo ======================================================================
cd /d "%~dp0"
set PYTHONUNBUFFERED=1

if exist ".venv\Scripts\python.exe" (
    echo Using Virtual Environment Python: .venv\Scripts\python.exe
    ".venv\Scripts\python.exe" "src\run_experiment_a_v2.py"
) else if exist "C:\Users\maheit\anaconda3\python.exe" (
    echo Using Anaconda Python: C:\Users\maheit\anaconda3\python.exe
    "C:\Users\maheit\anaconda3\python.exe" "src\run_experiment_a_v2.py"
) else (
    echo Using System Python:
    python "src\run_experiment_a_v2.py"
)

echo.
echo ======================================================================
echo Execution Finished!
echo ======================================================================
pause
