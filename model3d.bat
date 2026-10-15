@echo off
rem Make a 3D model of an apple tree from its summer photo, and open it in the browser.
rem   Drag a summer photo onto this file (or several, or any of a dataset tree's files),
rem   or double-click it and type a tree id such as R12N02.
rem It runs "python -m treeskel.model3d build" from this folder, with the .venv's Python if there is one.
setlocal
cd /d "%~dp0"
set "PY=python"
if exist ".venv\Scripts\python.exe" set "PY=.venv\Scripts\python.exe"

if "%~1"=="" goto ask
"%PY%" -m treeskel.model3d build %*
echo.
pause
goto :eof

:ask
echo Type a tree id (for example R12N02) or the path of a summer photo, then Enter.
echo Press Enter on its own to stop.
:again
set "TREE="
set /p "TREE=Tree: "
if not defined TREE goto :eof
"%PY%" -m treeskel.model3d build %TREE%
echo.
goto again
