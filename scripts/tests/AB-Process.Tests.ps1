$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot '..\AB-Process.ps1')

function Assert-Equal($Actual, $Expected) {
    if ($Actual -ne $Expected) { throw "Expected '$Expected', got '$Actual'" }
}

$ps = (Get-Process -Id $PID).Path
$natural = Start-Process $ps -ArgumentList '-NoProfile', '-Command', 'Start-Sleep -Milliseconds 500; exit 7' -PassThru -WindowStyle Hidden
try {
    [void]$natural.Handle
    if (!$natural.WaitForExit(10000)) { throw 'Natural-exit fixture timed out' }
    Assert-Equal (Get-ABProcessExit $natural $false) 'exited; exit-code:0x00000007'
    Assert-Equal (Stop-ABTrackedProcess $natural) 'exited; exit-code:0x00000007'
} finally {
    if (!$natural.HasExited) { $natural.Kill() }
    $natural.Dispose()
}

$stopped = Start-Process $ps -ArgumentList '-NoProfile', '-Command', 'Start-Sleep -Seconds 30' -PassThru -WindowStyle Hidden
try {
    [void]$stopped.Handle
    $status = Stop-ABTrackedProcess $stopped
    if ($status -notmatch '^stop-requested; exit-code:0x[0-9A-F]{8}$') { throw "Unexpected stop status: $status" }
} finally {
    if (!$stopped.HasExited) { $stopped.Kill() }
    $stopped.Dispose()
}

$unknown = [pscustomobject]@{ HasExited = $true; ExitCode = $null }
$unknown | Add-Member ScriptMethod Refresh { }
Assert-Equal (Get-ABProcessExit $unknown $false) 'exited; exit-code:unavailable'
$running = [pscustomobject]@{ HasExited = $false }
$running | Add-Member ScriptMethod Refresh { }
Assert-Equal (Get-ABProcessExit $running $false) 'still-running; exit-code-unavailable'
Write-Output 'AB-Process synthetic tests passed'
