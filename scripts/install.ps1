$ErrorActionPreference = "Stop"

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Host "Installing uv..."
    Invoke-RestMethod https://astral.sh/uv/install.ps1 | Invoke-Expression
    $env:Path = "$env:USERPROFILE\.local\bin;$env:Path"
}

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    throw "Could not find uv after installation. Open a new terminal and run this installer again."
}

Write-Host "Installing OpenCLI..."
uv tool install --upgrade opencli
Write-Host "OpenCLI installed. Run: opencli"
