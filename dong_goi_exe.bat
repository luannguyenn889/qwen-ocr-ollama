@echo off
chcp 65001 >nul
title Đóng gói Qwen OCR sang .exe
echo ========================================================
echo   BẮT ĐẦU ĐÓNG GÓI QWEN OCR APP SANG FILE .EXE
echo ========================================================
echo.

cd /d "%~dp0"

echo [1/3] Kích hoạt môi trường ảo...
call .venv\Scripts\activate.bat

echo [2/3] Kiểm tra & cài đặt PyInstaller...
pip install --quiet pyinstaller

echo [3/3] Tiến hành biên dịch và đóng gói ứng dụng (sẽ mất 2-3 phút)...
pyinstaller --noconfirm --clean QwenOCR.spec

if %ERRORLEVEL% equ 0 (
    echo.
    echo ========================================================
    echo   [THÀNH CÔNG] Ứng dụng đã được đóng gói tại:
    echo   dist\QwenOCR_App\QwenOCR_App.exe
    echo ========================================================
    echo.
) else (
    echo.
    echo ========================================================
    echo   [LỖI] Có lỗi xảy ra trong quá trình đóng gói!
    echo ========================================================
    echo.
)

pause
