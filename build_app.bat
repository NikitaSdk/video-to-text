@echo off
cd /d "%~dp0"
".venv\Scripts\python.exe" -m PyInstaller --noconfirm --onefile --noconsole --icon "icon.ico" --name VideoText launcher.py
if errorlevel 1 (
    echo Сборка не удалась.
    pause
    exit /b 1
)
copy /Y "dist\VideoText.exe" "%USERPROFILE%\Desktop\VideoText.exe"
echo.
echo Готово. На рабочем столе файл VideoText.exe
echo Чтобы сменить значок, замените icon.ico в этой папке и запустите этот файл снова.
pause
