# A/B queue for cool_fox variants on Vlad's PC (RTX 5060 Ti, 5700X, 32 GB).
# Runs each variant twice (run 1 warms its shader cache, run 2 is measured),
# reaches the prison scene by itself (VICTORY_KEYS presses gamepad A = Enter),
# samples the build's own frame counter (window title) once a second, takes a
# screenshot for picture checks, and closes the game by its own process id.
# Output: B:\cool_fox\ab\<stamp>\summary.csv, <variant>-run<n>.csv, .png, .log
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File B:\cool_fox\AB-Queue.ps1
#   ... -Only base,w64      (subset)   -Runs 1   (no warm-up run)   -Measure 90
param(
    [string] $Root = 'B:\cool_fox',
    [string[]] $Only,
    [int] $Runs = 2,
    [int] $Measure = 90,          # seconds measured once the scene is reached
    [int] $LoadTimeout = 600,     # seconds allowed to reach the scene
    [switch] $NoWarm              # skip `victory warm` between the two runs
)
$Warm = -not $NoWarm
$ErrorActionPreference = 'Continue'   # git writes progress to stderr (PS 5.1 would stop)
$stamp = Get-Date -f yyyyMMdd-HHmm
$out = Join-Path $Root "ab\$stamp"
New-Item -ItemType Directory -Force $out | Out-Null
$summary = Join-Path $out 'summary.csv'
'variant,run,samples,median_ms,p95_ms,max_ms,seconds_over_250ms,median_fps,draws_median,commit_gb_end,load_s,result,prep_ms,gpu_wait_ms,copyback_ms,card_idle_ms,card_idle_behind_us_ms,stood_for_card_ms' | Set-Content $summary

# name, checkout (relative to $Root; 'src' = base), git ref to check out (null: leave as is), extra env
$variants = @(
    @{ Name = 'base';      Dir = 'src'; Ref = $null; Env = @{} },
    @{ Name = 'base-env';  Dir = 'src'; Ref = $null; Env = @{ VICTORY_FILL_TITLE = '1'; VICTORY_CARD_BUFFERS_MIN_MB = '1'; VICTORY_CARD_BUFFERS_MB = '1536' } },
    @{ Name = 'r2';        Dir = 'r2';  Ref = 'origin/vlad/audit-round2'; Env = @{} },
    @{ Name = 'np';        Dir = 'np';  Ref = 'origin/vlad/nvidia-picks'; Env = @{} },
    @{ Name = 'w64';       Dir = 'w64'; Ref = 'origin/vlad/nvidia-picks-wave64'; Env = @{} },
    @{ Name = 'w64-env';   Dir = 'w64'; Ref = 'origin/vlad/nvidia-picks-wave64'; Env = @{ VICTORY_FILL_TITLE = '1'; VICTORY_CARD_BUFFERS_MIN_MB = '1'; VICTORY_CARD_BUFFERS_MB = '1536' } },
    # fxceless 06702fe VICTORY_SETTLE_ROOM (-11.5 ms of the preparing thread
    # there) is twice the lists in the air and 8x the merged runs; cool_fox
    # already merges 4096 runs with card steps, so the env alone is the same.
    @{ Name = 'w64-room';  Dir = 'w64'; Ref = 'origin/vlad/nvidia-picks-wave64'; Env = @{ VICTORY_LATE_WAIT_LISTS = '512' } },
    # Diagnostics build of w64: where the CPU time goes (VICTORY_PROFILE), in
    # the scene of the measured run. Its fps is not comparable with the others.
    @{ Name = 'w64-prof';  Dir = 'w64p'; Ref = 'origin/vlad/nvidia-picks-wave64'; Env = @{}; Diag = $true }
)
# Objects, not hashtables: Sort-Object/-Unique in PS 5.1 cannot see hashtable keys.
$variants = @($variants | ForEach-Object { [pscustomobject]$_ })
if ($Only) { $variants = $variants | Where-Object { $_.Name -in $Only } }

Add-Type -AssemblyName System.Windows.Forms, System.Drawing
function Shot([string] $path) {
    $b = [System.Windows.Forms.Screen]::PrimaryScreen.Bounds
    $bmp = New-Object System.Drawing.Bitmap $b.Width, $b.Height
    $g = [System.Drawing.Graphics]::FromImage($bmp)
    $g.CopyFromScreen($b.Location, [System.Drawing.Point]::Empty, $b.Size)
    $bmp.Save($path); $g.Dispose(); $bmp.Dispose()
}
# The build's own frame line (every second, in the run log): of a frame, ms
# preparing submissions, waiting for the GPU, copying back; and the card's
# clock: idle, idle while our submitting thread worked, that thread waiting.
# Medians of the last $n lines (the measured part: the game is closed then).
function Split([string] $log, [int] $n) {
    if (!$log -or !(Test-Path $log)) { return ',,,,,' }
    $lines = @(Select-String -Path $log -Pattern 'ms/frame \(|the card''s clock per frame' | ForEach-Object { $_.Line })
    $frame = @($lines | Where-Object { $_ -match 'ms/frame \(' } | Select-Object -Last $n)
    $clock = @($lines | Where-Object { $_ -match 'card''s clock' } | Select-Object -Last $n)
    $med = {
        param($rows, [string] $re)
        $v = @($rows | ForEach-Object { if ($_ -match $re) { [double]$matches[1] } } | Sort-Object)
        if ($v.Count) { $v[[int]($v.Count / 2)] } else { '' }
    }
    @(
        (& $med $frame 'ms/frame \(([\d.]+) preparing'),
        (& $med $frame '; ([\d.]+) on the\s+GPU'),
        (& $med $frame 'GPU; ([\d.]+) copying back'),
        (& $med $clock 'nothing of ours queued ([\d.]+) ms'),
        (& $med $clock 'queued [\d.]+ ms .*?, ([\d.]+) ms of that'),
        (& $med $clock 'waiting for the card ([\d.]+) ms')
    ) -join ','
}
function Pct($sorted, [double] $q) { if (!$sorted.Count) { return '' }; $sorted[[Math]::Min($sorted.Count - 1, [int]($sorted.Count * $q))] }

# 1. Checkouts and builds (each its own target dir)
$src = Join-Path $Root 'src'
git -C $src fetch origin 2>&1 | Out-Null
foreach ($v in $variants | Where-Object { $_.Ref } | Sort-Object Dir -Unique) {
    $dir = Join-Path $Root $v.Dir
    if (!(Test-Path (Join-Path $dir '.git'))) { git -C $src worktree add --detach $dir $v.Ref 2>&1 | Out-Null }
    else { git -C $dir checkout --detach $v.Ref 2>&1 | Out-Null }
    $sha = git -C $dir rev-parse --short HEAD
    "$($v.Dir) at $sha" | Add-Content (Join-Path $out 'builds.txt')
    # CoolFox.ps1 -Variant <dir> takes src\target-<dir>\release\victory.exe and
    # copies it to <dir>\target\release; the copy is made here too, so either
    # version of CoolFox.ps1 finds it.
    $tdir = Join-Path $src "target-$($v.Dir)"
    $prev = $env:CARGO_TARGET_DIR; $env:CARGO_TARGET_DIR = $tdir
    Push-Location $dir
    $how = if ($v.Diag) { '--profile diag --features diagnostics' } else { '--release' }
    try { cmd /c "cargo build $how --locked -p victory > `"$out\build-$($v.Dir).log`" 2>&1" } finally { Pop-Location; $env:CARGO_TARGET_DIR = $prev }
    $code = $LASTEXITCODE
    "  build exit $code" | Add-Content (Join-Path $out 'builds.txt')
    if ($code -eq 0) {
        New-Item -ItemType Directory -Force (Join-Path $dir 'target\release') | Out-Null
        $built = Join-Path $tdir $(if ($v.Diag) { 'diag' } else { 'release' })
        Copy-Item (Join-Path $built 'victory.exe') (Join-Path $dir 'target\release\victory.exe') -Force
        if ($v.Diag) { Copy-Item (Join-Path $built 'victory.pdb') $out -ErrorAction SilentlyContinue; Copy-Item (Join-Path $built 'victory.exe') $out }
    } else {
        # A failed build must not leave an older binary to be measured.
        Remove-Item (Join-Path $dir 'target\release\victory.exe') -ErrorAction SilentlyContinue
    }
}

# 2. Runs
$streak = 0; $loadS = ''
$keys = (5..120 | Where-Object { $_ % 5 -eq 0 } | ForEach-Object { "$($_):Enter" }) -join ','
foreach ($v in $variants) {
    $exePath = (Join-Path $Root "$($v.Dir)\target\release\victory.exe").ToLower()
    if (!(Test-Path $exePath)) { "$($v.Name),,,,,,,,,,,no exe" | Add-Content $summary; continue }
    $lastLoad = $null
    for ($run = 1; $run -le $Runs; $run++) {
        if (Get-Process victory -ErrorAction SilentlyContinue) { "$($v.Name),$run,,,,,,,,,,another victory.exe running: skipped" | Add-Content $summary; continue }
        $saved = @{}
        foreach ($k in @($v.Env.Keys) + 'VICTORY_KEYS') { $saved[$k] = [Environment]::GetEnvironmentVariable($k) }
        foreach ($k in $v.Env.Keys) { Set-Item "Env:$k" $v.Env[$k] }
        $env:VICTORY_KEYS = $keys
        # The warm-up run writes down the pipelines it makes; `victory warm`
        # then makes them ahead for the measured run (zone-entry stutters).
        $saved['VICTORY_PIPELINE_BOOK'] = $env:VICTORY_PIPELINE_BOOK
        $env:VICTORY_PIPELINE_BOOK = $(if ($run -eq 1 -and $Runs -gt 1) { '1' } else { $null })
        # The profile: 60 s of the last run, from 30 s after the scene was
        # reached in the run before (a warm run loads no slower).
        $saved['VICTORY_PROFILE'] = $env:VICTORY_PROFILE
        $env:VICTORY_PROFILE = $null
        if ($v.Diag -and $run -eq $Runs) {
            $from = $(if ($lastLoad) { $lastLoad + 30 } else { 240 })
            $env:VICTORY_PROFILE = "$from,$($from + 60),10,$out\$($v.Name)-profile.txt"
        }
        $argList = @('-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', (Join-Path $Root 'CoolFox.ps1'))
        if ($v.Dir -ne 'src') { $argList += @('-Variant', $v.Dir) }
        $t0 = Get-Date
        $runner = Start-Process powershell -ArgumentList $argList -WorkingDirectory $Root -PassThru -WindowStyle Minimized
        foreach ($k in $saved.Keys) { [Environment]::SetEnvironmentVariable($k, $saved[$k]) }
        $game = $null
        while (!$game -and ((Get-Date) - $t0).TotalSeconds -lt 180) {
            Start-Sleep 2
            $game = Get-Process victory -ErrorAction SilentlyContinue | Where-Object { $_.Path -and $_.Path.ToLower() -eq $exePath } | Select-Object -First 1
        }
        if (!$game) {
            if (!$runner.HasExited) { Stop-Process -Id $runner.Id -ErrorAction SilentlyContinue }
            "$($v.Name),$run,,,,,,,,,,game did not start" | Add-Content $summary; continue
        }
        $csv = Join-Path $out "$($v.Name)-run$run.csv"
        't,fps,ms,draws,commit_gb' | Set-Content $csv
        $scene = $null; $result = 'ok'; $measured = 0
        while ($true) {
            Start-Sleep 1
            $game.Refresh()
            if ($game.HasExited) { $result = "exited $($game.ExitCode)"; break }
            $t = $game.MainWindowTitle
            if ($t -match '([\d.]+) fps ([\d.]+) ms/frame (\d+) draws') {
                $ms = [double]$matches[2]; $draws = [int]$matches[3]
                "$((Get-Date).ToString('HH:mm:ss')),$($matches[1]),$ms,$draws,$([math]::Round($game.PagedMemorySize64 / 1GB, 2))" | Add-Content $csv
                # The scene: thousands of draws at under 200 ms a frame for 20 s in a row.
                if (!$scene) {
                    if ($draws -gt 3000 -and $ms -lt 200) { $streak++ } else { $streak = 0 }
                    if ($streak -ge 20) { $scene = Get-Date; $loadS = [int]($scene - $t0).TotalSeconds; Shot (Join-Path $out "$($v.Name)-run$run-a.png") }
                } else { $measured++ }
            }
            if (!$scene -and ((Get-Date) - $t0).TotalSeconds -gt $LoadTimeout) { $result = 'scene not reached'; break }
            if ($scene -and $measured -ge $Measure) {
                # The profile is written when its minute ends: wait for it (3 min at most).
                $prof = "$out\$($v.Name)-profile.txt"
                if ($v.Diag -and $run -eq $Runs -and !(Test-Path $prof) -and $measured -lt $Measure + 180) { continue }
                break
            }
        }
        if (!$game.HasExited) { Shot (Join-Path $out "$($v.Name)-run$run-b.png"); Stop-Process -Id $game.Id; [void]$game.WaitForExit(60000); Start-Sleep 5 }
        if (!$runner.HasExited) { Stop-Process -Id $runner.Id -ErrorAction SilentlyContinue }
        if ($run -eq 1 -and $Runs -gt 1 -and $Warm) {
            # Same cache folder as CoolFox.ps1 gives this variant.
            $cache = "$env:LOCALAPPDATA\VIctory\" + $(if ($v.Dir -eq 'src') { 'cache-coolfox' } else { "cache-coolfox-$($v.Dir)" })
            $prevCache = $env:VICTORY_CACHE_DIR; $env:VICTORY_CACHE_DIR = $cache
            Push-Location (Join-Path $Root $v.Dir)
            $tw = Get-Date
            try { cmd /c "`"$exePath`" warm --used > `"$out\$($v.Name)-warm.log`" 2>&1" } finally { Pop-Location; $env:VICTORY_CACHE_DIR = $prevCache }
            "$($v.Name) warm exit $LASTEXITCODE in $([int]((Get-Date) - $tw).TotalSeconds) s" | Add-Content (Join-Path $out 'builds.txt')
        }
        $rows = Import-Csv $csv
        $m = @($rows | Select-Object -Last $measured)
        $msS = @($m | ForEach-Object { [double]$_.ms } | Sort-Object)
        $dS = @($m | ForEach-Object { [double]$_.draws } | Sort-Object)
        $med = Pct $msS 0.5
        $fps = if ($med) { [math]::Round(1000 / $med, 1) } else { '' }
        $end = if ($rows) { $rows[-1].commit_gb } else { '' }
        $log = Get-ChildItem "$Root\logs\run-*.log" -ErrorAction SilentlyContinue | Where-Object { $_.LastWriteTime -gt $t0 } |
            Sort-Object LastWriteTime | Select-Object -Last 1
        $split = Split $(if ($log) { $log.FullName }) ([Math]::Max($measured, 1))
        "$($v.Name),$run,$($msS.Count),$med,$(Pct $msS 0.95),$(if ($msS.Count) { $msS[-1] }),$(@($msS | Where-Object { $_ -gt 250 }).Count),$fps,$(Pct $dS 0.5),$end,$loadS,$result,$split" | Add-Content $summary
        $lastLoad = $(if ($loadS) { [int]$loadS } else { $null }); $loadS = ''; $streak = 0
        Start-Sleep 10
    }
}
Get-ChildItem $Root\logs\run-* | Where-Object { $_.LastWriteTime -gt (Get-Date).AddHours(-3) } | Copy-Item -Destination $out
Write-Host "Done: $summary"
Get-Content $summary
