$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $projectRoot

py -m PyInstaller `
  --noconfirm `
  --clean `
  --onefile `
  --console `
  --paths src `
  --name SheetVideoToPdf `
  windows_app.py

if ($LASTEXITCODE -ne 0) {
  exit $LASTEXITCODE
}

Write-Host ""
Write-Host "Built dist\SheetVideoToPdf.exe"
