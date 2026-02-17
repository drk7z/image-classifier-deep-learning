param(
    [Parameter(Mandatory = $true)]
    [string]$Username,

    [Parameter(Mandatory = $true)]
    [securestring]$Password
)

$root = Split-Path -Parent $PSScriptRoot
$target = Join-Path $root "deploy/nginx/.htpasswd"

$bstr = [System.Runtime.InteropServices.Marshal]::SecureStringToBSTR($Password)
$plainPassword = [System.Runtime.InteropServices.Marshal]::PtrToStringBSTR($bstr)

Write-Host "Gerando arquivo .htpasswd em: $target"

docker run --rm --entrypoint htpasswd httpd:2.4-alpine -nbB $Username $plainPassword | Out-File -Encoding ascii $target

[System.Runtime.InteropServices.Marshal]::ZeroFreeBSTR($bstr)

if ($LASTEXITCODE -ne 0) {
    Write-Error "Falha ao gerar .htpasswd. Verifique se o Docker está ativo."
    exit 1
}

Write-Host "Arquivo .htpasswd gerado com sucesso."
