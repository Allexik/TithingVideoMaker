param(
    [switch]$Clean
)

$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root

function Get-RunningPackagedApps {
    $distRoot = Join-Path $root "dist"
    Get-Process -ErrorAction SilentlyContinue |
        Where-Object {
            $_.ProcessName -in @("TithingVideoMaker", "TithingVideoMakerUI") -and
            $_.Path -like "$distRoot*"
        }
}

$runningApps = @(Get-RunningPackagedApps)
if ($runningApps.Count -gt 0) {
    $processList = $runningApps | ForEach-Object { "$($_.ProcessName) (PID $($_.Id))" }
    throw "Close running packaged apps before rebuilding: $($processList -join ', ')"
}

if ($Clean) {
    Remove-Item build, dist -Recurse -Force -ErrorAction SilentlyContinue
}

function Move-TopLevelDataDirs {
    param(
        [Parameter(Mandatory = $true)]
        [string]$AppName
    )

    $appDir = Join-Path $root "dist\$AppName"
    $internalDir = Join-Path $appDir "_internal"

    foreach ($name in @("assets", "backgrounds")) {
        $from = Join-Path $internalDir $name
        $to = Join-Path $appDir $name
        if (Test-Path $from) {
            if (Test-Path $to) {
                Remove-Item $to -Recurse -Force
            }
            Move-Item $from $to
        }
    }
}

function Build-App {
    param(
        [Parameter(Mandatory = $true)]
        [string]$AppName,
        [Parameter(Mandatory = $true)]
        [string]$EntryPoint,
        [Parameter(Mandatory = $true)]
        [string]$UiMode
    )

    poetry run pyinstaller `
        --noconfirm `
        --clean `
        --name $AppName `
        --onedir `
        --contents-directory _internal `
        $UiMode `
        --add-data "assets;assets" `
        --add-data "backgrounds;backgrounds" `
        --copy-metadata imageio `
        --copy-metadata imageio-ffmpeg `
        --copy-metadata moviepy `
        --copy-metadata proglog `
        --collect-all imageio_ffmpeg `
        --collect-submodules moviepy `
        --collect-submodules proglog `
        $EntryPoint

    Move-TopLevelDataDirs -AppName $AppName
}

Build-App -AppName "TithingVideoMaker" -EntryPoint "main.py" -UiMode "--console"
Build-App -AppName "TithingVideoMakerUI" -EntryPoint "launch_ui.py" -UiMode "--windowed"
