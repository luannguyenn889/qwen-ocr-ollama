$cert = Get-ChildItem -Path "Cert:\CurrentUser\My" -CodeSigningCert | Select-Object -First 1
if (-not $cert) {
    $cert = New-SelfSignedCertificate -Type CodeSigningCert -Subject "CN=QwenOCR Developer" -CertStoreLocation "Cert:\CurrentUser\My"
}
Set-AuthenticodeSignature -FilePath "$PSScriptRoot\..\dist\QwenOCR_App\QwenOCR_App.exe" -Certificate $cert
