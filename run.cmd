@echo off
setlocal
set "PYTHONIOENCODING=utf-8"
if not exist "%~dp0runtime-python.txt" (
  echo Set runtime-python.txt to the Python executable path first.
  exit /b 1
)
set /p "taskPython=" < "%~dp0runtime-python.txt"
if exist "%~dp0runtime-storage.txt" set /p "MOVIE_STORAGE_DIR=" < "%~dp0runtime-storage.txt"
if not exist "%taskPython%" (
  echo Python executable from runtime-python.txt not found.
  exit /b 1
)
if "%~1"=="" (
  echo Usage: run.cmd -m scripts.ask_movies "Movie question"
  exit /b 0
)
pushd "%~dp0"
"%taskPython%" %*
set "taskExit=%ERRORLEVEL%"
popd
exit /b %taskExit%
