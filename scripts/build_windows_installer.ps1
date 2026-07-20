[CmdletBinding()]
param(
    [Parameter()]
    [string]$Version = '',

    [Parameter()]
    [string]$Python = 'python',

    [Parameter()]
    [string]$SignCertificateThumbprint = $env:TRAVELMOVIEAI_SIGN_CERTIFICATE
)

$ErrorActionPreference = 'Stop'
$repository = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
if ([string]::IsNullOrWhiteSpace($Version)) {
    $projectFile = Join-Path $repository 'pyproject.toml'
    $Version = (& $Python -c "import pathlib,sys,tomllib; print(tomllib.loads(pathlib.Path(sys.argv[1]).read_text(encoding='utf-8'))['project']['version'])" $projectFile)
}
if ($Version -notmatch '^\d+\.\d+\.\d+([.-][0-9A-Za-z.-]+)?$') {
    throw "Invalid application version: $Version"
}
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

$isccCommand = Get-Command 'ISCC.exe' -ErrorAction SilentlyContinue
$isccPath = if ($null -ne $isccCommand) { $isccCommand.Source } else { $null }
if ([string]::IsNullOrWhiteSpace($isccPath)) {
    $candidate = Join-Path ${env:ProgramFiles(x86)} 'Inno Setup 6\ISCC.exe'
    if (Test-Path -LiteralPath $candidate -PathType Leaf) {
        $isccPath = (Get-Item -LiteralPath $candidate).FullName
    }
}
if ([string]::IsNullOrWhiteSpace($isccPath)) {
    throw 'Inno Setup 6 was not found. Install it or add ISCC.exe to PATH.'
}

$sourceRoot = Join-Path $repository 'dist\TravelMovieAI'
$installerOutput = Join-Path $repository 'dist\installer'
$installerName = "TravelMovieAI-$Version-setup.exe"
$expectedInstaller = Join-Path $installerOutput $installerName
if (Test-Path -LiteralPath $expectedInstaller -PathType Leaf) {
    Remove-Item -LiteralPath $expectedInstaller -Force
}
& $isccPath "/DAppVersion=$Version" "/DSourceRoot=$sourceRoot" "/DInstallerOutput=$installerOutput" (Join-Path $repository 'installer\TravelMovieAI.iss')
if ($LASTEXITCODE -ne 0) { throw 'Inno Setup failed to build the installer.' }

if (-not (Test-Path -LiteralPath $expectedInstaller -PathType Leaf)) {
    throw "Inno Setup completed without producing an installer executable at the expected path: $installerName"
}
$installer = Get-Item -LiteralPath $expectedInstaller
if (-not [string]::IsNullOrWhiteSpace($SignCertificateThumbprint)) {
    $signToolCommand = Get-Command 'signtool.exe' -ErrorAction SilentlyContinue
    $signTool = if ($null -ne $signToolCommand) { $signToolCommand.Source } else { $null }
    if ([string]::IsNullOrWhiteSpace($signTool)) {
        throw 'Signing was requested, but signtool.exe was not found.'
    }
    & $signTool sign /sha1 $SignCertificateThumbprint /fd SHA256 /tr 'http://timestamp.digicert.com' /td SHA256 $installer.FullName
    if ($LASTEXITCODE -ne 0) { throw "Authenticode signing failed: $($installer.Name)" }
}
$hash = (Get-FileHash -LiteralPath $installer.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
"$hash  $($installer.Name)" | Set-Content -LiteralPath "$($installer.FullName).sha256" -Encoding ascii
$installer.FullName
