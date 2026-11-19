# FLAG 2027 - encoder fine-tuning on this PC's GPU (both variants, one after the other).
# Run in PowerShell from this folder:   powershell -ExecutionPolicy Bypass -File .\run.ps1
# If it stops with "CUDA out of memory", run:   powershell -ExecutionPolicy Bypass -File .\run.ps1 -P 48
# A variant that already finished is skipped when you run it again.
param([int]$P = 64)
$Base = $PSScriptRoot; $VP = Join-Path $Base "venv\Scripts\python.exe"
$env:IN_ROOT = Join-Path $Base "input"; $env:OUT_ROOT = Join-Path $Base "out"; $env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"
$v = '{"0": {"name": "sg50", "same_g": 0.5, "steps": 4000, "warm": 800, "eval_every": 500, "seed": 0, "p": PP}, ' +
     '"1": {"name": "sg90", "same_g": 0.9, "steps": 4000, "warm": 800, "eval_every": 500, "seed": 0, "p": PP}}'
$env:VARIANTS = $v.Replace("PP", "$P")
Write-Host "Logs are also saved in $($env:OUT_ROOT)\log_sg50.txt and log_sg90.txt"
& $VP -u (Join-Path $Base "xmodal_ft_local.py")
if ($LASTEXITCODE -eq 0) { Write-Host "`nFINISHED - now run:  powershell -ExecutionPolicy Bypass -File .\upload.ps1" -ForegroundColor Green }
