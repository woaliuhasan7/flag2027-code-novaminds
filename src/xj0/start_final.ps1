# FLAG 2027 - FINAL run on ALL 70 FLAG ids (the validated run-2 recipe: N-way loss + query pooling), IN THE BACKGROUND.
#   powershell -ExecutionPolicy Bypass -File .\start_final.ps1            (uses step 3500, the best step of both validation runs)
# Closing VS Code or losing RustDesk does not stop it. After a PC restart run the same command again (it continues from its checkpoint).
# Check progress:  powershell -ExecutionPolicy Bypass -File .\status.ps1        Stop:  powershell -ExecutionPolicy Bypass -File .\stop.ps1
param([int]$Step = 3500)
$Base = $PSScriptRoot
$running = Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object { $_.CommandLine -like "*xft_full.py*" }
if ($running) { Write-Host "A run is already going - check it with:  powershell -ExecutionPolicy Bypass -File .\status.ps1" -ForegroundColor Yellow; exit 1 }
$null = New-Item -ItemType Directory -Force (Join-Path $Base "out")
$stamp = Get-Date -Format "MMdd_HHmm"; $out = Join-Path $Base "out\console_$stamp.txt"; $err = Join-Path $Base "out\console_${stamp}_errors.txt"
$cmd = "& '$Base\run2.ps1' -Final $Step -Nway -QPool -NwayN 0"
Start-Process powershell -ArgumentList "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", $cmd -WindowStyle Hidden -RedirectStandardOutput $out -RedirectStandardError $err
Start-Sleep -Seconds 5
Write-Host "Final run started in the background (final_all70_nway_qp, step $Step). About 2.5-3 hours." -ForegroundColor Green
