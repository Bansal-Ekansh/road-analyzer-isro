@echo off
echo ============================================
echo  Route Resilience Analyzer - ISRO BAH 2026
echo ============================================
cd /d "%~dp0"

REM Try system python first, then Windows Store python
set ST=streamlit
where streamlit >nul 2>&1
if %errorlevel% neq 0 (
    set ST=%LOCALAPPDATA%\Packages\PythonSoftwareFoundation.Python.3.13_qbz5n2kfra8p0\LocalCache\local-packages\Python313\Scripts\streamlit.exe
)

echo Starting app at http://localhost:8501
"%ST%" run app/main.py --server.headless false --browser.gatherUsageStats false
