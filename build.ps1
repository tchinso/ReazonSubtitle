[CmdletBinding()]
param(
    [string]$PythonExe = ""
)

$ErrorActionPreference = "Stop"
$projectRoot = $PSScriptRoot
$distRoot = Join-Path $projectRoot "dist"
$distApp = Join-Path $distRoot "ReazonSubtitle"
$sourceAssets = Join-Path $projectRoot "assets"

if (-not $PythonExe) {
    $mekiPython = Join-Path (Split-Path $projectRoot -Parent) "MekiCopy\.build-python\python.exe"
    if (Test-Path -LiteralPath $mekiPython) {
        $PythonExe = $mekiPython
    } else {
        $command = Get-Command python -ErrorAction SilentlyContinue
        if ($command) {
            $PythonExe = $command.Source
        }
    }
}
if (-not $PythonExe -or -not (Test-Path -LiteralPath $PythonExe)) {
    throw "Python 3.12 build environment was not found. Pass -PythonExe explicitly."
}

$requiredAssets = @(
    "ffmpeg\ffmpeg.exe",
    "ffmpeg\ffprobe.exe",
    "vad\silero_vad.onnx",
    "reazonspeech-ja\tokens.txt",
    "reazonspeech-ja\encoder-epoch-99-avg-1.int8.onnx",
    "reazonspeech-ja\decoder-epoch-99-avg-1.onnx",
    "reazonspeech-ja\joiner-epoch-99-avg-1.int8.onnx",
    "tchinso\Hy-MT2-1.8B-onnx-q4f16\onnx\model_q4f16.onnx",
    "onnx-community\HY-MT1.5-1.8B-ONNX\onnx\model_q4.onnx_data",
    "hytrans-runtime\worker.html",
    "hytrans-runtime\worker.js",
    "hytrans-runtime\transformers.min.js",
    "hytrans-runtime\wasm\ort-wasm-simd-threaded.asyncify.wasm"
)
foreach ($relative in $requiredAssets) {
    $path = Join-Path $sourceAssets $relative
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
        throw "Required asset is missing: $path"
    }
}

Push-Location $projectRoot
try {
    & $PythonExe -m PyInstaller --noconfirm --clean ".\ReazonSubtitle.spec"
    if ($LASTEXITCODE -ne 0) {
        throw "PyInstaller failed with exit code $LASTEXITCODE"
    }
} finally {
    Pop-Location
}

$targetAssets = Join-Path $distApp "assets"
New-Item -ItemType Directory -Path $targetAssets -Force | Out-Null
$sourceRootResolved = (Resolve-Path -LiteralPath $sourceAssets).Path
$distAppResolved = (Resolve-Path -LiteralPath $distApp).Path
if (-not $distAppResolved.StartsWith((Resolve-Path -LiteralPath $distRoot).Path + "\", [System.StringComparison]::OrdinalIgnoreCase)) {
    throw "Refusing to publish outside the dist directory: $distAppResolved"
}

Get-ChildItem -LiteralPath $sourceRootResolved -Recurse -File | ForEach-Object {
    $relative = $_.FullName.Substring($sourceRootResolved.Length).TrimStart("\")
    $target = Join-Path $targetAssets $relative
    $targetDirectory = Split-Path $target -Parent
    New-Item -ItemType Directory -Path $targetDirectory -Force | Out-Null
    if (Test-Path -LiteralPath $target) {
        Remove-Item -LiteralPath $target -Force
    }
    try {
        New-Item -ItemType HardLink -Path $target -Target $_.FullName -ErrorAction Stop | Out-Null
    } catch {
        Copy-Item -LiteralPath $_.FullName -Destination $target -Force
    }
}

$exe = Join-Path $distApp "ReazonSubtitle.exe"
if (-not (Test-Path -LiteralPath $exe -PathType Leaf)) {
    throw "Built executable is missing: $exe"
}

$process = Start-Process -FilePath $exe -ArgumentList "--self-test" -PassThru -Wait -WindowStyle Hidden
if ($process.ExitCode -ne 0) {
    throw "Frozen executable self-test failed with exit code $($process.ExitCode)"
}

$assetBytes = (Get-ChildItem -LiteralPath $targetAssets -Recurse -File | Measure-Object Length -Sum).Sum
Write-Host "Build complete: $exe"
Write-Host ("Published assets: {0:N2} GiB" -f ($assetBytes / 1GB))
