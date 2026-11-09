@echo off
rem baslat.bat - Agent Smith (Mars Turizm chatbot) baslatici / yeniden baslatici
rem
rem Ne yapar:
rem   1. 8501'de calisan eski app.py Streamlit surecini durdurur (yeniden baslatma).
rem   2. Tailscale Funnel'i 8501'e acar (zaten aciksa zarari yok).
rem   3. Streamlit'i bu pencerede baslatir. Pencereyi kapatmak = botu durdurmak.
rem
rem Funnel'i kapatmak icin:  tailscale funnel --https=443 off

title Agent Smith - 8501
cd /d "%~dp0"

echo.
echo   Agent Smith  -  http://127.0.0.1:8501
echo.

echo   [1/3] Eski surec durduruluyor...
powershell -NoProfile -Command "$p = @(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { $_.CommandLine -like '*streamlit*app.py*' }); $p | ForEach-Object { Stop-Process -Id $_.ProcessId -Force }; Write-Host ('         durdurulan: ' + $p.Count)"

echo   [2/3] Tailscale Funnel aciliyor...
tailscale funnel --bg 8501 >nul 2>&1
if errorlevel 1 (
    echo         UYARI: Funnel acilamadi. Bot yalniz bu bilgisayarda calisir.
) else (
    powershell -NoProfile -Command "$u = (tailscale funnel status) -match '^https://' | Select-Object -First 1; if (-not $u) { $u = 'adres icin: tailscale funnel status' }; Write-Host ('         Funnel acik: ' + $u)"
)

echo   [3/3] Streamlit baslatiliyor - ilk acilista model isinir...
echo.
"C:\Users\Sarp\anaconda3\python.exe" -m streamlit run app.py

echo.
echo   Streamlit kapandi.
pause
