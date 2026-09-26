param(
    [Parameter(Mandatory=$true)]
    [ValidateSet("clinc150","mixatis","mixsnips","all")]
    [string]$Dataset,
    [ValidateSet("calib","test","all")]
    [string]$Split = "all",
    [int]$Limit = 0
)
$ErrorActionPreference = "Continue"
$ROOT = "E:\PY-Project\WWW2027\code\intent-conformal"
Set-Location $ROOT
$env:HF_ENDPOINT = "https://hf-mirror.com"
$PY = ".\.venv\Scripts\python.exe"
$TS = Get-Date -Format "yyyyMMdd_HHmmss"
$RESULTS_DIR = "$ROOT\results"
New-Item -ItemType Directory -Force -Path $RESULTS_DIR | Out-Null

$DATASETS = if ($Dataset -eq "all") { @("clinc150","mixatis","mixsnips") } else { @($Dataset) }
$SPLITS = if ($Split -eq "all") { @("calib","test") } else { @($Split) }

foreach ($ds in $DATASETS) {
    Write-Host "`n========== Dataset: $ds ==========" -ForegroundColor Cyan
    foreach ($sp in $SPLITS) {
        $predFile = "$ROOT\data\predictions\$ds\$sp.jsonl"
        if (Test-Path $predFile) {
            $n = (Get-Content $predFile -Encoding UTF8 | Measure-Object -Line).Lines
            Write-Host "[skip] $ds/$sp already has $n predictions" -ForegroundColor Yellow
            continue
        }
        Write-Host "[run]  LLM inference: $ds/$sp" -ForegroundColor Green
        $limitArg = if ($Limit -gt 0) { "--limit $Limit" } else { "" }
        & $PY scripts/03_run_llm.py --dataset $ds --split $sp $limitArg 2>&1 | Tee-Object -Variable llmOut
    }
    $calibFile = "$ROOT\data\predictions\$ds\calib.jsonl"
    $testFile  = "$ROOT\data\predictions\$ds\test.jsonl"
    if ((Test-Path $calibFile) -and (Test-Path $testFile)) {
        Write-Host "[run]  Calibration + Eval: $ds" -ForegroundColor Green
        $outFile = "$RESULTS_DIR\${ds}_$TS.txt"
        & $PY scripts/04_calibrate_and_eval.py --dataset $ds 2>&1 | Tee-Object -FilePath $outFile
        Write-Host "[saved] $outFile" -ForegroundColor Gray
    }
}
Write-Host "`n========== Done ==========" -ForegroundColor Cyan
