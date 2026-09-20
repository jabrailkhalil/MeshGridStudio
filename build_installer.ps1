$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$version = "1.2.0"
$versionFile = Join-Path $projectRoot "mesh_grid_studio_version.txt"
if (Test-Path -LiteralPath $versionFile) {
    $match = Select-String -Path $versionFile -Pattern "FileVersion', '([0-9.]+)'" | Select-Object -First 1
    if ($match -and $match.Matches.Groups.Count -gt 1) {
        $version = $match.Matches[0].Groups[1].Value
    }
}

$appDir = Join-Path $projectRoot "output\app\MeshGridStudio"
$installerDir = Join-Path $projectRoot "output\installer"
$scriptDir = Join-Path $projectRoot "build\installer"
if (-not (Test-Path -LiteralPath $appDir)) {
    throw "Onedir build not found at $appDir; run build_exe.ps1 first"
}
New-Item -ItemType Directory -Force -Path $installerDir | Out-Null
New-Item -ItemType Directory -Force -Path $scriptDir | Out-Null

$issPath = Join-Path $scriptDir "MeshGridStudio.iss"
$appDirEscaped = $appDir.Replace("\", "\\")

$iss = @"
#define MyAppName "Mesh Grid Studio"
#define MyAppVersion "$version"
#define MyAppPublisher "D. E. Khalilov"
#define MyAppExeName "MeshGridStudio.exe"

[Setup]
AppId={{4F7A9C31-2B5E-4D6A-9C8B-0E1F3A5B7D29}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
DefaultDirName={autopf}\MeshGridStudio
DefaultGroupName={#MyAppName}
DisableProgramGroupPage=yes
OutputDir=$installerDir
OutputBaseFilename=MeshGridStudio-$version-Setup
Compression=lzma2
SolidCompression=yes
WizardStyle=modern
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
UninstallDisplayName={#MyAppName}

[Tasks]
Name: "desktopicon"; Description: "Create a desktop shortcut"; GroupDescription: "Additional icons:"

[Files]
Source: "$appDirEscaped\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "Launch {#MyAppName}"; Flags: nowait postinstall skipifsilent
"@
Set-Content -LiteralPath $issPath -Value $iss -Encoding UTF8

$iscc = @(
    "$env:LOCALAPPDATA\Programs\Inno Setup 6\ISCC.exe",
    "C:\Program Files (x86)\Inno Setup 6\ISCC.exe",
    "C:\Program Files\Inno Setup 6\ISCC.exe"
) | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
if (-not $iscc) {
    throw "Inno Setup (ISCC.exe) not found; install it first"
}

& $iscc $issPath
if ($LASTEXITCODE -ne 0) {
    throw "Inno Setup exited with code $LASTEXITCODE"
}

Get-ChildItem -LiteralPath $installerDir -Filter "*.exe" |
    Select-Object FullName, Length, LastWriteTime