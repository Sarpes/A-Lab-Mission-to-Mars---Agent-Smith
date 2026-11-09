@echo off
rem durdur.bat - Agent Smith (Mars Turizm chatbot) kapatici
rem
rem Ne yapar:
rem   1. app.py'yi calistiran Streamlit surecini durdurur (Janus TR 8502'ye dokunmaz).
rem   2. Tailscale Funnel'in herkese acik 443 adresini kapatir (n8n 5678 tailnet kaydi kalir).
rem   3. Ollama modellerini GPU belleginden bosaltir (Ollama'nin kendisi acik kalir).
rem   4. Kontrol: 8501'de dinleyen kalmadi mi, Funnel kapandi mi.

title Agent Smith - durdur
cd /d "%~dp0"

echo.
echo   Agent Smith kapatiliyor
echo.

echo   [1/3] Streamlit durduruluyor...
powershell -NoProfile -Command "$p = @(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { $_.CommandLine -like '*streamlit*app.py*' }); $p | ForEach-Object { Stop-Process -Id $_.ProcessId -Force }; Write-Host ('         durdurulan: ' + $p.Count)"

echo   [2/3] Tailscale Funnel kapatiliyor...
tailscale funnel --https=443 off >nul 2>&1
if errorlevel 1 (
    echo         Funnel zaten kapaliydi ya da kapatilamadi.
) else (
    echo         Funnel kapandi.
)

echo   [3/3] Modeller bellekten bosaltiliyor...
ollama stop gemma4:latest >nul 2>&1
ollama stop bge-m3 >nul 2>&1
echo         tamam.

echo.
echo   Kontrol:
powershell -NoProfile -Command "Start-Sleep -Seconds 2; $n = @(Get-NetTCPConnection -State Listen -LocalPort 8501 -ErrorAction SilentlyContinue).Count; if ($n -eq 0) { Write-Host '     8501: kapali' } else { Write-Host '     8501: HALA ACIK - elle kontrol et' }; $f = (tailscale funnel status) -join ' '; if ($f -match '8501') { Write-Host '     Funnel: HALA ACIK - elle kontrol et' } else { Write-Host '     Funnel: kapali' }"
echo.
pause
