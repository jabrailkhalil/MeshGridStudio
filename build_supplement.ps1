$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$outputRoot = Join-Path $projectRoot "output"
$stagingDirectory = Join-Path $outputRoot "supplementary-materials"
$archivePath = Join-Path $outputRoot "article-supplementary-materials.zip"
$temporaryArchive = Join-Path $outputRoot "article-supplementary-materials.new.zip"

$resolvedOutput = [System.IO.Path]::GetFullPath($outputRoot)
$resolvedStaging = [System.IO.Path]::GetFullPath($stagingDirectory)
if (-not $resolvedStaging.StartsWith(
        $resolvedOutput + [System.IO.Path]::DirectorySeparatorChar,
        [System.StringComparison]::OrdinalIgnoreCase
    )) {
    throw "Unsafe staging path: $resolvedStaging"
}

function Copy-RequiredFile {
    param(
        [Parameter(Mandatory = $true)][string]$Source,
        [Parameter(Mandatory = $true)][string]$Destination
    )

    if (-not (Test-Path -LiteralPath $Source -PathType Leaf)) {
        throw "Required file not found: $Source"
    }
    $destinationDirectory = Split-Path -Parent $Destination
    if ($destinationDirectory) {
        New-Item -ItemType Directory -Force -Path $destinationDirectory |
            Out-Null
    }
    Copy-Item -LiteralPath $Source -Destination $Destination -Force
}

if (Test-Path -LiteralPath $stagingDirectory) {
    Remove-Item -LiteralPath $stagingDirectory -Recurse -Force
}
New-Item -ItemType Directory -Path $stagingDirectory | Out-Null

$rootFiles = @(
    "article.tex",
    "AUDIT.md",
    "build_exe.ps1",
    "build_supplement.ps1",
    "mesh_grid_studio_version.txt",
    "mesh_gui.py",
    "mesh_gui_model.py",
    "mesh_methods.py",
    "package_supplement.py",
    "README.md",
    "requirements-build.txt",
    "requirements.txt",
    "SUPPLEMENTARY.md",
    "test_mesh_gui.py",
    "test_mesh_methods.py",
    "tishkin_grid.tex"
)
foreach ($relativePath in $rootFiles) {
    Copy-RequiredFile `
        -Source (Join-Path $projectRoot $relativePath) `
        -Destination (Join-Path $stagingDirectory $relativePath)
}

$generatedSource = Join-Path $outputRoot "generated"
$generatedDestination = Join-Path $stagingDirectory "output\generated"
if (-not (Test-Path -LiteralPath $generatedSource -PathType Container)) {
    throw "Generated results directory not found: $generatedSource"
}
$generatedFiles = @(
    "adaptive_mu0_control.json",
    "mesh_overlays.pdf",
    "mesh_overlays.png",
    "meshes_arch.pdf",
    "meshes_arch.png",
    "meshes_circle.pdf",
    "meshes_circle.png",
    "meshes_square.pdf",
    "meshes_square.png",
    "results.csv",
    "results_table.tex",
    "run_metadata.json"
)
foreach ($fileName in $generatedFiles) {
    Copy-RequiredFile `
        -Source (Join-Path $generatedSource $fileName) `
        -Destination (Join-Path $generatedDestination $fileName)
}

Copy-RequiredFile `
    -Source (Join-Path $outputRoot "pdf\article.pdf") `
    -Destination (Join-Path $stagingDirectory "output\pdf\article.pdf")
Copy-RequiredFile `
    -Source (Join-Path $outputRoot "app\ui-preview-fast-exe.png") `
    -Destination (
        Join-Path $stagingDirectory "output\app\ui-preview-fast-exe.png"
    )
Copy-RequiredFile `
    -Source (Join-Path $projectRoot "docs\ui-preview.png") `
    -Destination (Join-Path $stagingDirectory "docs\ui-preview.png")

if (Test-Path -LiteralPath $temporaryArchive) {
    Remove-Item -LiteralPath $temporaryArchive -Force
}
& python (Join-Path $projectRoot "package_supplement.py") `
    $stagingDirectory $temporaryArchive
if ($LASTEXITCODE -ne 0) {
    throw "Cross-platform ZIP creation failed with exit code $LASTEXITCODE"
}

if (Test-Path -LiteralPath $archivePath) {
    Remove-Item -LiteralPath $archivePath -Force
}
Move-Item -LiteralPath $temporaryArchive -Destination $archivePath

Get-Item -LiteralPath $archivePath |
    Select-Object FullName, Length, LastWriteTime
