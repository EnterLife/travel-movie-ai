[CmdletBinding()]
param(
    [Parameter()]
    [ValidatePattern('^\d+\.\d+\.\d+([.-][0-9A-Za-z.-]+)?$')]
    [string]$Version = '0.1.0',

    [Parameter()]
    [string]$Python = 'python'
)

$ErrorActionPreference = 'Stop'
$repository = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$builderEnvironment = Join-Path $repository '.cache\installer-venv'
$builderPython = Join-Path $builderEnvironment 'Scripts\python.exe'

if (-not (Test-Path -LiteralPath $builderPython -PathType Leaf)) {
    & $Python -m venv $builderEnvironment
    if ($LASTEXITCODE -ne 0) { throw 'Could not create the installer build environment.' }
}

& $builderPython -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { throw 'Could not update pip in the installer environment.' }
& $builderPython -m pip install -e "${repository}[desktop,installer]"
if ($LASTEXITCODE -ne 0) { throw 'Could not install desktop installer dependencies.' }

Push-Location $repository
try {
    & $builderPython -m PyInstaller --noconfirm --clean 'installer\travelmovieai.spec'
    if ($LASTEXITCODE -ne 0) { throw 'PyInstaller failed to build the desktop distribution.' }
}
finally {
    Pop-Location
}

$iscc = Get-Command 'ISCC.exe' -ErrorAction SilentlyContinue
if ($null -eq $iscc) {
    $candidate = Join-Path ${env:ProgramFiles(x86)} 'Inno Setup 6\ISCC.exe'
    if (Test-Path -LiteralPath $candidate -PathType Leaf) {
        $iscc = Get-Item -LiteralPath $candidate
    }
}
if ($null -eq $iscc) {
    throw 'Inno Setup 6 was not found. Install it or add ISCC.exe to PATH.'
}

$sourceRoot = Join-Path $repository 'dist\TravelMovieAI'
$installerOutput = Join-Path $repository 'dist\installer'
& $iscc.Source "/DAppVersion=$Version" "/DSourceRoot=$sourceRoot" "/DInstallerOutput=$installerOutput" (Join-Path $repository 'installer\TravelMovieAI.iss')
if ($LASTEXITCODE -ne 0) { throw 'Inno Setup failed to build the installer.' }

Get-ChildItem -LiteralPath $installerOutput -Filter '*.exe' | Select-Object -ExpandProperty FullName
