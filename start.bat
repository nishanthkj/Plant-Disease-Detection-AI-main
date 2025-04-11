@echo off
echo 🛠️ Starting setup...

:: Step 1: Check if venv exists
if exist venv (
    echo 🔁 Virtual environment already exists. Skipping creation.
) else (
    echo 🧪 Creating virtual environment...
    python -m venv venv
)

:: Step 2: Activate virtual environment
call venv\Scripts\activate

:: Step 3: Install dependencies
echo 📦 Installing Python packages...
pip install -r requirements.txt

:: Step 4: Handle .env and API key
if exist .env (
    echo ⚠️  .env file already exists.
    set /p CHANGEKEY=❓ Do you want to change the API key? (Y/N): 
    if /I "%CHANGEKEY%"=="Y" (
        set /p APIKEY=🔑 Enter your new Gemini API key: 
        echo GEMINI_API_KEY=%APIKEY%> .env
        echo ✅ API key updated in .env!
    ) else (
        echo ⏩ Skipping API key update.
    )
) else (
    set /p APIKEY=🔑 Enter your Gemini API key: 
    echo GEMINI_API_KEY=%APIKEY%> .env
    echo ✅ .env file created!
)

:: Step 5: Run the app
echo 🚀 Running app...
python app.py

pause
