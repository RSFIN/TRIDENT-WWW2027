# P4_finish_and_continue.ps1 - seed123 eval -> seed2024 full pipeline -> restore seed42 baseline
# NOTE: ASCII only (PowerShell 5.1 reads no-BOM UTF-8 as ANSI; non-ASCII breaks parsing)
$ErrorActionPreference = "Continue"
$ROOT = "E:\PY-Project\WWW2027\code\intent-conformal"
$py = "$ROOT\.venv\Scripts\python.exe"
cd $ROOT

$env:HF_HUB_OFFLINE = "1"
$env:HF_ENDPOINT = "https://hf-mirror.com"
$env:TRANSFORMERS_OFFLINE = "1"

Write-Output ("[t0] seed123 eval start " + (Get-Date).ToString("yyyy-MM-dd HH:mm:ss"))

# 1) seed=123 unified eval (20_ablation_unified, K=1/K_r=1, pred-subdir s123)
foreach ($ds in @("clinc150", "mixatis", "mixsnips")) {
    Write-Output ("[eval123] " + $ds)
    & $py scripts\20_ablation_unified.py --dataset $ds --k-max 1 --kr 1 --pred-subdir "s123" 2>&1 | Tee-Object -FilePath "results\p4_seed123.txt" | Out-Null
    if ($LASTEXITCODE -ne 0) { Write-Output "[FAIL] eval123 $ds exit=$LASTEXITCODE"; exit 1 }
}
Write-Output ("[t1] seed123 eval done " + (Get-Date).ToString("yyyy-MM-dd HH:mm:ss"))

# 2) seed=2024 full pipeline (inline: 01 split/KB -> 02 index -> 03 LLM calib+test -> 20 eval)
$Seed = 2024
$cfg = "configs\default.yaml"
$c = [IO.File]::ReadAllText($cfg)
$c = [regex]::Replace($c, '(?m)^seed:\s*\d+', "seed: $Seed")
[IO.File]::WriteAllText($cfg, $c, (New-Object Text.UTF8Encoding $false))
Write-Output "[cfg] seed set to 2024"

& $py scripts\01_download_data.py | Select-Object -Last 2
if ($LASTEXITCODE -ne 0) { Write-Output "[FAIL] 01 seed2024"; exit 1 }
Write-Output "[01] split/KB done"

& $py scripts\02_build_index.py | Select-Object -Last 2
if ($LASTEXITCODE -ne 0) { Write-Output "[FAIL] 02 seed2024"; exit 1 }
Write-Output "[02] index done"

foreach ($ds in @("clinc150", "mixatis", "mixsnips")) {
    Write-Output ("[03] " + $ds + " calib")
    & $py scripts\03_run_llm.py --dataset $ds --split calib --out-subdir "s2024" --workers 6 | Select-Object -Last 2
    if ($LASTEXITCODE -ne 0) { Write-Output "[FAIL] 03 calib seed2024 $ds"; exit 1 }
    Write-Output ("[03] " + $ds + " test")
    & $py scripts\03_run_llm.py --dataset $ds --split test --out-subdir "s2024" --workers 6 | Select-Object -Last 2
    if ($LASTEXITCODE -ne 0) { Write-Output "[FAIL] 03 test seed2024 $ds"; exit 1 }
}
Write-Output "[03] LLM done"

foreach ($ds in @("clinc150", "mixatis", "mixsnips")) {
    Write-Output ("[eval2024] " + $ds)
    & $py scripts\20_ablation_unified.py --dataset $ds --k-max 1 --kr 1 --pred-subdir "s2024" 2>&1 | Tee-Object -FilePath "results\p4_seed2024.txt" | Out-Null
    if ($LASTEXITCODE -ne 0) { Write-Output "[FAIL] eval2024 $ds"; exit 1 }
}
Write-Output ("[t2] seed2024 done " + (Get-Date).ToString("yyyy-MM-dd HH:mm:ss"))

# 3) restore seed42 baseline (config + processed + indexes)
Copy-Item "data\_p4_backup\default.yaml" "configs\default.yaml" -Force
Copy-Item "data\_p4_backup\processed\*" "data\processed\" -Recurse -Force
Copy-Item "data\_p4_backup\indexes\*" "data\indexes\" -Recurse -Force
Write-Output "[t3] seed42 baseline restored"

# 4) append progress log
Add-Content -Path "results\p4_progress_log.txt" -Value ("{0} | P4 COMPLETE: seed123 eval + seed2024 done + seed42 restored" -f (Get-Date).ToString("yyyy-MM-dd HH:mm:ss")) -Encoding utf8
Write-Output ("[ALL DONE] " + (Get-Date).ToString("yyyy-MM-dd HH:mm:ss"))
