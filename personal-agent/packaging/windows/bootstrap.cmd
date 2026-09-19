@echo off
setlocal EnableDelayedExpansion
title PersonalOS Agent - Install

REM ===========================================================================
REM  PersonalOS Agent - clone and deploy in one step.
REM
REM  Download this file anywhere and double-click it, or run:
REM      bootstrap.cmd
REM      bootstrap.cmd C:\Code            (where to clone)
REM ===========================================================================

set "REPO=https://github.com/pushkarkumar25dmba175-rgb/stakpak-agent"
set "BRANCH=claude/stoic-lovelace-3fzaxx"

if "%~1"=="" (set "TARGET=%USERPROFILE%\Code") else (set "TARGET=%~1")
set "CLONE=%TARGET%\stakpak-agent"

echo.
echo  ============================================================
echo   PersonalOS Agent - install
echo  ============================================================
echo.
echo   Cloning into: %CLONE%
echo.

git --version >nul 2>&1 || (
    echo  [X] Git is not installed or not on PATH.
    echo      Install it from https://git-scm.com/download/win and try again.
    exit /b 1
)

if not exist "%TARGET%" mkdir "%TARGET%" || (echo  [X] Could not create %TARGET%. & exit /b 1)

if exist "%CLONE%\.git" (
    echo  Repository already present - updating it.
    pushd "%CLONE%" || exit /b 1
    git fetch origin "%BRANCH%" || (echo  [X] Could not fetch. & popd & exit /b 1)
    git checkout "%BRANCH%" || (echo  [X] Could not check out %BRANCH%. & popd & exit /b 1)
    git pull origin "%BRANCH%" || (echo  [X] Could not pull. & popd & exit /b 1)
    popd
) else (
    git clone --branch "%BRANCH%" "%REPO%" "%CLONE%" || (echo  [X] Clone failed. & exit /b 1)
)

echo.
echo  Repository ready. Running deployment...
echo.

call "%CLONE%\personal-agent\packaging\windows\deploy.cmd" %*
exit /b %errorlevel%
