param(
    [Parameter(Mandatory = $true)][int]$Seed,
    [string]$Suffix = ""
)
# P4 单 seed 全流程：改 seed -> 01 划分/重建 KB -> 02 重建索引 -> 03 LLM(calib+test, 隔离目录) -> 20 评估
cd "E:\PY-Project\WWW2027\code\intent-conformal"
$py = "E:\PY-Project\WWW2027\code\intent-conformal\.venv\Scripts\python.exe"

# 0) HF 离线（01 的 load_dataset 与 02 的 embedder 均需本地缓存）
$env:HF_HUB_OFFLINE = "1"
$env:HF_ENDPOINT = "https://hf-mirror.com"
$env:TRANSFORMERS_OFFLINE = "1"

# 1) config 改 seed（UTF-8 无 BOM 写回）
$cfg = "configs\default.yaml"
$c = [IO.File]::ReadAllText($cfg)
$c = [regex]::Replace($c, '(?m)^seed:\s*\d+', "seed: $Seed")
[IO.File]::WriteAllText($cfg, $c, (New-Object Text.UTF8Encoding $false))
Write-Output "[$Seed] config seed set"

# 2) 01 数据划分 + KB
& $py scripts\01_download_data.py | Select-Object -Last 3
if ($LASTEXITCODE -ne 0) { throw "01 failed seed=$Seed" }

# 3) 02 重建索引（HF 离线）
& $py scripts\02_build_index.py | Select-Object -Last 3
if ($LASTEXITCODE -ne 0) { throw "02 failed seed=$Seed" }

# 4) 03 LLM：calib + test，输出到 s{Seed} 隔离目录
foreach ($ds in @("clinc150", "mixatis", "mixsnips")) {
    & $py scripts\03_run_llm.py --dataset $ds --split calib --out-subdir "s$Seed" | Select-Object -Last 2
    if ($LASTEXITCODE -ne 0) { throw "03 calib failed seed=$Seed ds=$ds" }
    & $py scripts\03_run_llm.py --dataset $ds --split test --out-subdir "s$Seed" | Select-Object -Last 2
    if ($LASTEXITCODE -ne 0) { throw "03 test failed seed=$Seed ds=$ds" }
}
Write-Output "[$Seed] LLM done"

# 5) 20 评估（统一口径，主配置 K=1/K_r=1）
foreach ($ds in @("clinc150", "mixatis", "mixsnips")) {
    & $py scripts\20_ablation_unified.py --dataset $ds --k-max 1 --kr 1 --pred-subdir "s$Seed" 2>&1 | Tee-Object -FilePath "results\p4_seed${Seed}.txt" | Select-Object -Last 2
    if ($LASTEXITCODE -ne 0) { throw "20 eval failed seed=$Seed ds=$ds" }
}
Write-Output "[$Seed] P4 seed done"
