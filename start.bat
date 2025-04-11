@echo off
setlocal

:: Set the virtual environment directory
set VENV_DIR=venv

:: Step 1: Create Virtual Environment if it doesn't exist
if not exist "%VENV_DIR%\Scripts\python.exe" (
    echo Creating virtual environment...
    python -m venv "%VENV_DIR%"
)

:: Step 2: Install Dependencies
echo Installing dependencies...
"%VENV_DIR%\Scripts\python.exe" -m pip install --upgrade pip
"%VENV_DIR%\Scripts\python.exe" -m pip install -r requirements.txt

:: Step 3: Configure Gemini API Key
if not exist .env (
    set /p APIKEY=Enter your Gemini API key: 
    echo GEMINI_API_KEY=%APIKEY%> .env
    echo .env file created with your API key.
)

:: Step 4: Run the Application
echo Running the application...
"%VENV_DIR%\Scripts\python.exe" app.py

endlocal
pause
