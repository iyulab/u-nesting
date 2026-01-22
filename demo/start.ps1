# U-Nesting Demo Server Launcher
# Usage: .\start.ps1

$ErrorActionPreference = "Stop"
$devDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectRoot = Split-Path -Parent $devDir

Write-Host "U-Nesting Demo Server" -ForegroundColor Cyan
Write-Host "=====================" -ForegroundColor Cyan

# Build the benchmark binary (always rebuild to get latest changes)
Write-Host "Building u-nesting-benchmark (release)..." -ForegroundColor Yellow
Push-Location $projectRoot
try {
    cargo build --release -p u-nesting-benchmark
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Build failed!" -ForegroundColor Red
        exit 1
    }
    Write-Host "Build complete!" -ForegroundColor Green
} finally {
    Pop-Location
}

# Check if port 8888 is already in use
$existing = Get-NetTCPConnection -LocalPort 8888 -ErrorAction SilentlyContinue
if ($existing) {
    Write-Host "Port 8888 already in use. Stopping existing process..." -ForegroundColor Yellow
    $existing | ForEach-Object { Stop-Process -Id $_.OwningProcess -Force -ErrorAction SilentlyContinue }
    Start-Sleep -Seconds 1
}

# Start server
Write-Host "Starting server on http://localhost:8888" -ForegroundColor Green
Start-Process python -ArgumentList "$devDir\server.py" -WorkingDirectory $devDir -WindowStyle Hidden

Start-Sleep -Seconds 2

# Open browser
Write-Host "Opening browser..." -ForegroundColor Green
Start-Process "http://localhost:8888"

Write-Host ""
Write-Host "Demo ready! Workflow:" -ForegroundColor Cyan
Write-Host "  1. Click 'Load Samples' (42 manufacturing parts)"
Write-Host "  2. Click 'Random' to see chaos"
Write-Host "  3. Click 'Optimize' to run U-Nesting library"
Write-Host ""
Write-Host "Press Ctrl+C to stop server" -ForegroundColor Yellow

# Keep script running to show server output
try {
    while ($true) {
        Start-Sleep -Seconds 60
    }
} finally {
    Write-Host "Stopping server..." -ForegroundColor Yellow
    Get-Process python -ErrorAction SilentlyContinue | Where-Object { $_.Path -like "*python*" } | Stop-Process -Force -ErrorAction SilentlyContinue
}
