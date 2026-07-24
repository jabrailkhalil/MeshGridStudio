$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$outputDirectory = Join-Path $projectRoot "output\app"
$workDirectory = Join-Path $projectRoot "build\pyinstaller\work"
$specDirectory = Join-Path $projectRoot "build\pyinstaller"
$entryPoint = Join-Path $projectRoot "mesh_gui.py"
$versionFile = Join-Path $projectRoot "mesh_grid_studio_version.txt"
$legacyExecutable = Join-Path $outputDirectory "MeshGridStudio.exe"
$legacyDirectory = Join-Path $outputDirectory "legacy"

New-Item -ItemType Directory -Force -Path $outputDirectory | Out-Null
New-Item -ItemType Directory -Force -Path $workDirectory | Out-Null
New-Item -ItemType Directory -Force -Path $specDirectory | Out-Null
if (Test-Path -LiteralPath $legacyExecutable -PathType Leaf) {
    New-Item -ItemType Directory -Force -Path $legacyDirectory | Out-Null
    Move-Item -LiteralPath $legacyExecutable `
        -Destination (Join-Path $legacyDirectory "MeshGridStudio-onefile-slow.exe") `
        -Force
}

Push-Location $projectRoot
try {
    python -m PyInstaller `
        --noconfirm `
        --clean `
        --onedir `
        --contents-directory "runtime" `
        --windowed `
        --name "MeshGridStudio" `
        --version-file $versionFile `
        --distpath $outputDirectory `
        --workpath $workDirectory `
        --specpath $specDirectory `
        --hidden-import "matplotlib.backends.backend_tkagg" `
        --exclude-module "PyQt5" `
        --exclude-module "PyQt6" `
        --exclude-module "PySide2" `
        --exclude-module "PySide6" `
        --exclude-module "IPython" `
        --exclude-module "jupyter" `
        --exclude-module "nbformat" `
        --exclude-module "notebook" `
        --exclude-module "pygame" `
        --exclude-module "pandas" `
        --exclude-module "lxml" `
        --exclude-module "zmq" `
        $entryPoint

    if ($LASTEXITCODE -ne 0) {
        throw "PyInstaller завершился с кодом $LASTEXITCODE"
    }
}
finally {
    Pop-Location
}

$executable = Join-Path $outputDirectory "MeshGridStudio\MeshGridStudio.exe"
if (-not (Test-Path -LiteralPath $executable)) {
    throw "Сборка завершилась без ожидаемого файла $executable"
}

Get-Item -LiteralPath $executable |
    Select-Object FullName, Length, LastWriteTime
