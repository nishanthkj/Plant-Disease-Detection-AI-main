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

:: Step 4: Check if .env exists
if exist .env (
    echo ⚠️  .env file already exists. Skipping API key input.
) else (
    set /p APIKEY=🔑 Enter your Gemini API key: 
    echo GEMINI_API_KEY=%APIKEY%> .env
    echo ✅ .env file created!
)

:: Step 5: Run the app
echo 🚀 Running app...
python app.py

pause
