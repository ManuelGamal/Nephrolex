param(
  [switch]$SkipDense,
  [switch]$DoclingOcr,
  [string]$DenseModel = "BAAI/bge-m3",
  [int]$BatchSize = 4
)

$ErrorActionPreference = "Stop"
$env:DOCLING_INFERENCE_COMPILE_TORCH_MODELS = "false"
$env:TORCH_COMPILE_DISABLE = "1"
$env:TORCHDYNAMO_DISABLE = "1"

if (-not (Test-Path ".\.venv\Scripts\python.exe")) {
  python -m venv .venv
}

.\.venv\Scripts\python.exe -m pip install -r .\requirements.txt

$parseArgs = @(".\scripts\parse_guidelines.py", "--run-docling", "--docling-bin", ".\.venv\Scripts\docling.exe")
if ($DoclingOcr) {
  $parseArgs += "--docling-ocr"
}
.\.venv\Scripts\python.exe @parseArgs
.\.venv\Scripts\python.exe .\scripts\chunk_guidelines.py

if ($SkipDense) {
  .\.venv\Scripts\python.exe .\scripts\build_indexes.py --skip-dense
  .\.venv\Scripts\python.exe .\scripts\evaluate_retrieval.py --top-k 8
} else {
  .\.venv\Scripts\python.exe .\scripts\build_indexes.py --dense-model $DenseModel --batch-size $BatchSize
  .\.venv\Scripts\python.exe .\scripts\evaluate_retrieval.py --top-k 8 --dense-model $DenseModel
}
