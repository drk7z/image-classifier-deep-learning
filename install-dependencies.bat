@echo off
setlocal

cd /d "%~dp0"

if "%~1"=="-h" goto :usage
if "%~1"=="--help" goto :usage

if not exist "requirements.txt" (
    echo [ERRO] Arquivo requirements.txt nao encontrado na pasta do projeto.
    exit /b 1
)

if exist ".venv\Scripts\python.exe" (
    set "PYTHON_EXE=.venv\Scripts\python.exe"
) else (
    echo Ambiente virtual .venv nao encontrado. Criando com Python 3.11...
    py -3.11 -m venv .venv >nul 2>&1
    if errorlevel 1 (
        echo [ERRO] Falha ao criar .venv com "py -3.11".
        echo Verifique se o Python 3.11 esta instalado e no PATH.
        exit /b 1
    )
    set "PYTHON_EXE=.venv\Scripts\python.exe"
)

echo Atualizando pip...
"%PYTHON_EXE%" -m pip install --upgrade pip
if errorlevel 1 (
    echo [ERRO] Falha ao atualizar o pip.
    exit /b 1
)

echo Instalando dependencias de requirements.txt...
"%PYTHON_EXE%" -m pip install -r requirements.txt
if errorlevel 1 (
    echo [ERRO] Falha ao instalar dependencias.
    exit /b 1
)

echo.
echo [OK] Dependencias instaladas com sucesso.
echo Use: .venv\Scripts\python.exe app.py
exit /b 0

:usage
echo Uso:
echo   install-dependencies.bat
echo.
echo O script:
echo   1. Cria .venv com Python 3.11 (se nao existir)
echo   2. Atualiza pip
echo   3. Instala requirements.txt
exit /b 0

endlocal
