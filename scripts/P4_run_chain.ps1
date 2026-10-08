# P4_run_chain.ps1 — P4 seed=123 自动接力脚本
# 依次运行 03_run_llm.py 的剩余 split，每步完成后检查目标行数（达标即跳过，支持断点续跑）
# 用法: powershell -ExecutionPolicy Bypass -File scripts\P4_run_chain.ps1
$ErrorActionPreference = "Continue"
$ROOT = "E:\PY-Project\WWW2027\code\intent-conformal"
$py = "$ROOT\.venv\Scripts\python.exe"
$WORKERS = 6

$env:HF_HUB_OFFLINE = "1"
$env:HF_ENDPOINT = "https://hf-mirror.com"
$env:TRANSFORMERS_OFFLINE = "1"

# split -> (目标行数, 输出文件相对路径)
$steps = @(
  @{ds='clinc150'; split='calib'; target=3812;  out='data\predictions\clinc150_s123\calib.jsonl'},
  @{ds='clinc150'; split='test';  target=5500;  out='data\predictions\clinc150_s123\test.jsonl'},
  @{ds='mixatis';  split='calib'; target=3284;  out='data\predictions\mixatis_s123\calib.jsonl'},
  @{ds='mixatis';  split='test';  target=828;   out='data\predictions\mixatis_s123\test.jsonl'},
  @{ds='mixsnips'; split='calib'; target=9942;  out='data\predictions\mixsnips_s123\calib.jsonl'},
  @{ds='mixsnips'; split='test';  target=2199;  out='data\predictions\mixsnips_s123\test.jsonl'}
)

function Get-LineCount($path) {
  if (-not (Test-Path $path)) { return 0 }
  return (Get-Content $path -Encoding utf8 | Measure-Object -Line).Lines
}

Set-Location $ROOT

# 前置等待：若 clinc150/calib 仍在补跑（外部进程），先等它达标（最多 45 分钟）
$calibPath = "data\predictions\clinc150_s123\calib.jsonl"
$waited = 0
while ((Get-LineCount $calibPath) -lt 3812 -and $waited -lt 45) {
  Write-Output ("[wait] clinc150/calib {0}/3812 ..." -f (Get-LineCount $calibPath))
  Start-Sleep -Seconds 60
  $waited++
}

foreach ($st in $steps) {
  $done = Get-LineCount $st.out
  if ($done -ge $st.target) {
    Write-Output ("[skip] {0}/{1} already {2}/{3}" -f $st.ds,$st.split,$done,$st.target)
    continue
  }
  Write-Output ("[run]  {0}/{1} current={2} target={3} start={4}" -f $st.ds,$st.split,$done,$st.target,(Get-Date).ToString("HH:mm:ss"))
  $outLog = "results\p4_s123_logs\$($st.ds)_$($st.split).out"
  $errLog = "results\p4_s123_logs\$($st.ds)_$($st.split).err"
  & $py "scripts\03_run_llm.py" --dataset $st.ds --split $st.split --out-subdir s123 --workers $WORKERS *> $outLog
  # *> 会把 stderr 并入 out（PowerShell 5 对原生命令 stderr 有包装，但此处仅用于观察，不影响流程）
  $after = Get-LineCount $st.out
  Write-Output ("[end]  {0}/{1} after={2}/{3}" -f $st.ds,$st.split,$after,$st.target)
}
Write-Output ("[ALL DONE] " + (Get-Date).ToString("yyyy-MM-dd HH:mm:ss"))
