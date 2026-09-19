@echo off
setlocal EnableDelayedExpansion
title PersonalOS Agent - Deployment

REM ===========================================================================
REM  PersonalOS Agent - one-shot deployment for Windows.
REM
REM  Installs the agent into a virtual environment, sets up its home directory,
REM  and runs the built-in health check. Safe to re-run: it updates in place.
REM
REM  Usage:   deploy.cmd
REM           deploy.cmd C:\Path\To\Workspace
REM ===========================================================================

echo.
echo  ============================================================
echo   PersonalOS Agent - deployment
echo  ============================================================
echo.

REM ---- 0. Where things go ---------------------------------------------------
set "PROJECT_DIR=%~dp0..\.."
pushd "%PROJECT_DIR%" || (echo  [X] Cannot enter the project directory. & goto :fail)
set "PROJECT_DIR=%CD%"

if "%~1"=="" (
    set "WORKSPACE=%USERPROFILE%\PersonalOS"
) else (
    set "WORKSPACE=%~1"
)

echo  Project:   %PROJECT_DIR%
echo  Workspace: %WORKSPACE%
echo  Agent home: %USERPROFILE%\.personalos
echo.

REM ---- 1. Find a suitable Python -------------------------------------------
echo  [1/6] Looking for Python 3.12 or newer...
set "PY_CMD="

for %%V in (3.13 3.12) do (
    if not defined PY_CMD (
        py -%%V --version >nul 2>&1 && set "PY_CMD=py -%%V"
    )
)

if not defined PY_CMD (
    python --version >nul 2>&1 && (
        for /f "tokens=2" %%A in ('python --version 2^>^&1') do (
            for /f "tokens=1,2 delims=." %%M in ("%%A") do (
                if %%M GTR 3 set "PY_CMD=python"
                if %%M EQU 3 if %%N GEQ 12 set "PY_CMD=python"
            )
        )
    )
)

if not defined PY_CMD (
    echo.
    echo  [X] No Python 3.12+ found on this machine.
    echo.
    echo      Install it from https://www.python.org/downloads/
    echo      and tick "Add python.exe to PATH" during setup.
    echo      Then run this script again.
    goto :fail
)

for /f "tokens=*" %%A in ('%PY_CMD% --version 2^>^&1') do echo      Using %%A ^(%PY_CMD%^)

REM ---- 2. Virtual environment ----------------------------------------------
echo.
echo  [2/6] Creating the virtual environment...
if exist ".venv\Scripts\python.exe" (
    echo      Already present, reusing it.
) else (
    %PY_CMD% -m venv .venv || (echo  [X] Could not create the virtual environment. & goto :fail)
    echo      Created at %PROJECT_DIR%\.venv
)

set "VENV_PY=%PROJECT_DIR%\.venv\Scripts\python.exe"
set "AGENT=%PROJECT_DIR%\.venv\Scripts\agent.exe"

REM ---- 3. Dependencies ------------------------------------------------------
echo.
echo  [3/6] Installing dependencies ^(this takes a minute on a first run^)...
"%VENV_PY%" -m pip install --upgrade pip --quiet --disable-pip-version-check || (echo  [X] Could not upgrade pip. & goto :fail)
"%VENV_PY%" -m pip install --editable ".[all]" --quiet --disable-pip-version-check || (echo  [X] Dependency installation failed. See the output above. & goto :fail)
echo      Core, scheduler, watcher, dashboard and PDF support installed.

REM psutil is what lets the agent list processes on Windows.
"%VENV_PY%" -m pip install psutil --quiet --disable-pip-version-check >nul 2>&1
if errorlevel 1 (
    echo      [!] psutil did not install - process listing will be unavailable.
) else (
    echo      psutil installed - process listing enabled.
)

REM ---- 4. Workspace and agent home -----------------------------------------
echo.
echo  [4/6] Setting up the workspace...
if not exist "%WORKSPACE%" (
    mkdir "%WORKSPACE%" || (echo  [X] Could not create %WORKSPACE%. & goto :fail)
    echo      Created %WORKSPACE%
) else (
    echo      %WORKSPACE% already exists.
)

"%AGENT%" init --workspace "%WORKSPACE%" || (echo  [X] `agent init` failed. & goto :fail)

REM ---- 5. Health check ------------------------------------------------------
echo.
echo  [5/6] Running the health check...
echo.
"%AGENT%" doctor
if errorlevel 1 (
    echo.
    echo  [!] The health check reported a problem. The agent is installed, but
    echo      read the lines marked [X] above before relying on it.
) else (
    echo.
    echo      All checks passed.
)

REM ---- 6. Smoke test --------------------------------------------------------
echo.
echo  [6/6] Verifying the agent can actually do something...
echo.
"%AGENT%" ask "list %WORKSPACE%" --no-show-plan
if errorlevel 1 (
    echo  [!] The test task did not complete. See the output above.
) else (
    echo.
    echo      The agent read your workspace successfully.
)

REM ---- Done -----------------------------------------------------------------
echo.
echo  ============================================================
echo   Deployment complete.
echo  ============================================================
echo.
echo   Run the agent with:
echo.
echo     "%AGENT%" status
echo     "%AGENT%" ask "what is in my workspace?"
echo.
echo   Or activate the environment once and just type `agent`:
echo.
echo     call "%PROJECT_DIR%\.venv\Scripts\activate.bat"
echo     agent status
echo.
echo   To use a language model, set your key first ^(this session only^):
echo     set ANTHROPIC_API_KEY=sk-ant-...
echo   To keep it, use:  setx ANTHROPIC_API_KEY "sk-ant-..."
echo.
echo   Without a key the agent still runs - it plans from keywords
echo   and tells you so rather than pretending.
echo.
echo   Start with the computer:  packaging\windows\register-task.ps1
echo   Uninstall:                packaging\windows\uninstall.cmd
echo.
popd
endlocal
exit /b 0

:fail
echo.
echo  Deployment stopped. Nothing was left half-installed that a re-run
echo  will not fix - correct the problem above and run this script again.
echo.
popd 2>nul
endlocal
exit /b 1
