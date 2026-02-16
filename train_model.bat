@echo off
setlocal

cd /d "%~dp0"

if "%~1"=="-h" goto :usage
if "%~1"=="--help" goto :usage

if exist ".venv\Scripts\python.exe" (
    set "PYTHON_EXE=.venv\Scripts\python.exe"
) else (
    echo [ERRO] Ambiente virtual .venv nao encontrado.
    echo Crie com: py -3.11 -m venv .venv
    echo Instale dependencias com: .venv\Scripts\pip install -r requirements.txt
    exit /b 1
)

set "EPOCHS=%~1"
if "%EPOCHS%"=="" set "EPOCHS=50"

set "BATCH_SIZE=%~2"
if "%BATCH_SIZE%"=="" set "BATCH_SIZE=32"

set "DATA_DIR=%~3"
if "%DATA_DIR%"=="" set "DATA_DIR=data"

if not exist "%DATA_DIR%\train" (
    echo [ERRO] Pasta de treino nao encontrada: %DATA_DIR%\train
    goto :dataset_help
)

if not exist "%DATA_DIR%\validation" (
    echo [ERRO] Pasta de validacao nao encontrada: %DATA_DIR%\validation
    goto :dataset_help
)

echo Iniciando treino...
echo epochs=%EPOCHS% batch_size=%BATCH_SIZE% data_dir=%DATA_DIR%

"%PYTHON_EXE%" -c "from src.train import ImageClassifierTrainer; t=ImageClassifierTrainer(r'%DATA_DIR%'); t.train(epochs=%EPOCHS%, batch_size=%BATCH_SIZE%)"
exit /b %errorlevel%

:dataset_help
echo.
echo Estrutura esperada:
echo   data\train\cats
echo   data\train\dogs
echo   data\validation\cats
echo   data\validation\dogs
echo.
echo Exemplo:
echo   train_model.bat 10 16 data
exit /b 1

:usage
echo Uso:
echo   train_model.bat [epochs] [batch_size] [data_dir]
echo.
echo Exemplos:
echo   train_model.bat
echo   train_model.bat 10 16 data
echo.
echo Padroes:
echo   epochs=50, batch_size=32, data_dir=data
exit /b 0

endlocal