@echo off
setlocal
title PersonalOS Agent - Uninstall

REM ===========================================================================
REM  Removes the virtual environment and, if you confirm, the agent's home
REM  directory. Your workspace files are never touched.
REM ===========================================================================

set "PROJECT_DIR=%~dp0..\.."
pushd "%PROJECT_DIR%" || exit /b 1
set "PROJECT_DIR=%CD%"
set "HOME_DIR=%USERPROFILE%\.personalos"

echo.
echo  PersonalOS Agent - uninstall
echo.

REM Stop the scheduled task first, if it was registered.
schtasks /query /tn "PersonalOSAgent" >nul 2>&1 && (
    schtasks /delete /tn "PersonalOSAgent" /f >nul 2>&1
    echo  Removed the "PersonalOSAgent" scheduled task.
)

if exist "%PROJECT_DIR%\.venv" (
    rmdir /s /q "%PROJECT_DIR%\.venv"
    echo  Removed the virtual environment.
) else (
    echo  No virtual environment found.
)

echo.
echo  The agent's home directory holds your memory, skills, logs and trash:
echo      %HOME_DIR%
echo.
choice /c YN /n /m "  Delete it too? [Y/N] "
if errorlevel 2 goto :keep

if exist "%HOME_DIR%" (
    rmdir /s /q "%HOME_DIR%"
    echo  Removed %HOME_DIR%
) else (
    echo  Nothing to remove at %HOME_DIR%
)
goto :done

:keep
echo  Kept %HOME_DIR% - reinstalling will pick up where you left off.

:done
echo.
echo  Done. Your workspace files were not touched.
echo  Delete the repository folder itself to finish removing the code.
echo.
popd
endlocal
exit /b 0
