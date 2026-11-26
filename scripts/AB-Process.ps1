# Process tracking for AB-Queue. These helpers do not start the game.
function Get-ABProcessExit($Process, [bool] $StopRequested) {
    try {
        $Process.Refresh()
        if (!$Process.HasExited) { return 'still-running; exit-code-unavailable' }
        $code = $Process.ExitCode
        if ($null -eq $code) { $formatted = 'unavailable' }
        else { $formatted = '0x{0:X8}' -f [int]$code }
        $how = if ($StopRequested) { 'stop-requested' } else { 'exited' }
        return "$how; exit-code:$formatted"
    } catch {
        return 'exit-status-unavailable'
    }
}

function Stop-ABTrackedProcess($Process) {
    # Kill through the retained process object/handle, never by a possibly reused PID.
    # A successful request does not prove that the process did not exit naturally
    # at the same instant, hence the deliberately conservative status name.
    $requested = $false
    try {
        $Process.Refresh()
        if (!$Process.HasExited) {
            $Process.Kill()
            $requested = $true
            [void]$Process.WaitForExit(60000)
        }
    } catch {
        # A natural exit can race with Kill; report the actual status below.
    }
    Get-ABProcessExit $Process $requested
}
