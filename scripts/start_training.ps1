# One-click start, run by start_training.bat. Updates the code if asked, then
# hands over to `tmai start`, which reads tmai_settings.json and starts the
# plugins, the dashboard and training.

$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

$python = Join-Path $root ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) {
    Write-Host "No .venv found in $root. Do docs\SETUP_WINDOWS.md first." -ForegroundColor Red
    exit 1
}

$settings = Join-Path $root "tmai_settings.json"
if (-not (Test-Path $settings)) {
    Copy-Item (Join-Path $root "tmai_settings.example.json") $settings
    Write-Host "Created tmai_settings.json. Write your map name and buttons in it, save, then double-click start_training.bat again." -ForegroundColor Yellow
    Start-Process notepad.exe $settings
    exit 1
}

try {
    $s = Get-Content $settings -Raw -ErrorAction Stop | ConvertFrom-Json -ErrorAction Stop
} catch {
    Write-Host "tmai_settings.json is not valid JSON: $($_.Exception.Message)" -ForegroundColor Red
    exit 1
}

# Defaults to true, like tmai's own default for update_code.
# git writes progress to stderr, so no -ErrorAction Stop here.
if (($null -eq $s.update_code -or $s.update_code) -and -not (Get-Command git -ErrorAction SilentlyContinue)) {
    Write-Host "git not found; starting with the code you have." -ForegroundColor Yellow
} elseif ($null -eq $s.update_code -or $s.update_code) {
    Write-Host "Updating the code (git pull)..."
    $before = git rev-parse HEAD
    git pull --ff-only
    if ($LASTEXITCODE -ne 0) {
        Write-Host "git pull failed; starting with the code you have. See docs\TROUBLESHOOTING.md." -ForegroundColor Yellow
    } elseif ((git rev-parse HEAD) -ne $before) {
        Write-Host "New code pulled; reinstalling the Python package..."
        & $python -m pip install --quiet -e ".[game,dashboard,dev]"
        if ($LASTEXITCODE -ne 0) {
            Write-Host "pip install failed; see the messages above." -ForegroundColor Red
            exit 1
        }
    }
}

& $python -m tmai start
exit $LASTEXITCODE
