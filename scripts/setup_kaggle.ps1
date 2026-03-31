$ErrorActionPreference = "Stop"
python -m pip install kaggle
$kaggleDir = Join-Path $HOME ".kaggle"
New-Item -ItemType Directory -Force -Path $kaggleDir | Out-Null
Write-Host "Place kaggle.json in $kaggleDir before continuing."
kaggle datasets download -d ziya07/depvidmood-facial-expression-video-dataset -p data/raw/ --unzip
Write-Host "DepVidMood ready in data/raw/depvidmood-facial-expression-video-dataset"
