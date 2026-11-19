# FLAG 2027 - extra-seed runs for the ensemble, ONE AFTER THE OTHER, then upload.  Run in a normal PowerShell window (not VS Code):
#   powershell -ExecutionPolicy Bypass -File .\start_seeds.ps1
# Keep the window open. Losing RustDesk does not stop it. After a PC restart run the same command again: finished runs are skipped,
# a run that was cut off continues from its last checkpoint.
#   1. full_sg50_nway_qp_s1   (14 FLAG ids held out -> lets Claude check that the ensemble really helps)   ~3.4 h
#   2. full_sg50_nway_qp_s2   (same, seed 2)                                                               ~3.4 h
#   3. final_all70_nway_qp_s1 (all 70 ids, step 3500)                                                       ~1.7 h
#   4. final_all70_nway_qp_s2 (same, seed 2)                                                               ~1.7 h
#   5. upload of the FLAG feature files of all 4 runs -> shahiraalima/xft-seeds (private)
$Base = $PSScriptRoot
if (-not $env:KAGGLE_API_TOKEN) { $env:KAGGLE_API_TOKEN = Read-Host "Paste the Kaggle API token of the shahiraalima account (needed for the upload at the end)"; Clear-Host }
$running = Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object { $_.CommandLine -like "*xft_full.py*" }
if ($running) { Write-Host "A run is already going (python $($running[0].ProcessId)). Stop it first with stop.ps1" -ForegroundColor Yellow; exit 1 }
$runs = @(              # hashtable splatting (switches as $true) - array splatting would not bind -Nway etc. as named parameters
    @{ name = "full_sg50_nway_qp_s1";   args = @{ Nway = $true; QPool = $true; NwayN = 0; Seed = 1 } },
    @{ name = "full_sg50_nway_qp_s2";   args = @{ Nway = $true; QPool = $true; NwayN = 0; Seed = 2 } },
    @{ name = "final_all70_nway_qp_s1"; args = @{ Final = 3500; Nway = $true; QPool = $true; NwayN = 0; Seed = 1 } },
    @{ name = "final_all70_nway_qp_s2"; args = @{ Final = 3500; Nway = $true; QPool = $true; NwayN = 0; Seed = 2 } })
foreach ($r in $runs) {
    $log = Join-Path $Base ("out\xft\" + $r.name + "\log.txt")
    if ((Test-Path $log) -and ((Get-Content $log -Tail 1) -match "FINISHED")) { Write-Host "$($r.name): already finished" -ForegroundColor Green; continue }
    Write-Host "`n===== $($r.name)  (started $(Get-Date -Format 'dd MMM HH:mm'))" -ForegroundColor Cyan
    $a = $r.args; & (Join-Path $Base "run2.ps1") @a
    if (-not ((Test-Path $log) -and ((Get-Content $log -Tail 1) -match "FINISHED"))) { Write-Host "$($r.name) did not finish - run this script again (it continues from the checkpoint)" -ForegroundColor Red; exit 1 }
}
# upload: only the FLAG feature files + summaries (small)
$up = Join-Path $Base "seeds_upload"; Remove-Item $up -Recurse -Force -ErrorAction SilentlyContinue; $null = New-Item -ItemType Directory -Force $up
foreach ($r in $runs) {
    $src = Join-Path $Base ("out\xft\" + $r.name); $dst = Join-Path $up $r.name; $null = New-Item -ItemType Directory -Force (Join-Path $dst "flag")
    Copy-Item (Join-Path $src "flag\*.npz") (Join-Path $dst "flag"); Copy-Item (Join-Path $src "*.json") $dst; Copy-Item (Join-Path $src "log.txt") $dst
}
('{"title": "xft-seeds", "id": "shahiraalima/xft-seeds", "licenses": [{"name": "other"}]}') | Set-Content -Encoding ascii (Join-Path $up "dataset-metadata.json")
$KaggleExe = Join-Path $Base "venv\Scripts\kaggle.exe"
& $KaggleExe datasets create -p $up --dir-mode zip 2>&1 | Tee-Object -Variable o1 | Out-Host; $txt = ($o1 | Out-String)
if ($LASTEXITCODE -ne 0 -or $txt -match "error|already in use") { & $KaggleExe datasets version -p $up -m "update" --dir-mode zip 2>&1 | Tee-Object -Variable o2 | Out-Host; $txt = ($o2 | Out-String) }
if ($LASTEXITCODE -eq 0 -and $txt -notmatch "error") { Write-Host "`nALL DONE. UPLOADED shahiraalima/xft-seeds (private). Tell Claude it is done." -ForegroundColor Green }
else { Write-Host "`nRuns finished but the UPLOAD FAILED - send Claude the lines above" -ForegroundColor Red }
