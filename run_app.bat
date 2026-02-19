@echo off
setlocal

cd /d "%~dp0"

if exist ".venv\Scripts\python.exe" (
    set "PYTHON_EXE=.venv\Scripts\python.exe"
) else (
    echo [ERRO] Ambiente virtual .venv nao encontrado.
    echo Crie com: py -3.11 -m venv .venv
    echo Instale dependencias com: .venv\Scripts\pip install -r requirements.txt
    exit /b 1
)

"%PYTHON_EXE%" -c "import gradio" >nul 2>&1
if errorlevel 1 (
    echo Instalando dependencias do projeto...
    "%PYTHON_EXE%" -m pip install -r requirements.txt
    if errorlevel 1 exit /b 1
)

echo Iniciando app em http://localhost:7860
"%PYTHON_EXE%" app.py %*

endlocal