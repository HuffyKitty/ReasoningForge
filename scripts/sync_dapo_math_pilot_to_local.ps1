param([switch]$Wait)

$ErrorActionPreference = 'Stop'
$projectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$exportDir = Join-Path $projectRoot 'exports'
$runDir = Join-Path $projectRoot 'runs\qwen15b_dapo_math_17k_baseline_pilot'
$remoteHostName = 'autodl-reasoningforge'
$remoteRoot = '/root/autodl-tmp/ReasoningForge-next'
$archiveName = 'dapo_math_baseline_pilot.tar.gz'
$remoteArchive = "$remoteRoot/exports/$archiveName"
$archivePath = Join-Path $exportDir $archiveName
$shaPath = "$archivePath.sha256"
$modelMarker = Join-Path $exportDir 'dapo_math_baseline_pilot.local_sync_COMPLETE.json'
$evalMarker = Join-Path $exportDir 'dapo_math_baseline_dev500.local_sync_COMPLETE'
$logPath = Join-Path $exportDir 'dapo_math_local_sync.log'
$taskName = 'ReasoningForge-DapoMath-LocalSync'

New-Item -ItemType Directory -Force -Path $exportDir, (Join-Path $projectRoot 'runs') | Out-Null
$mutex = [System.Threading.Mutex]::new($false, 'Local\ReasoningForgeDapoMathLocalSync')
if (-not $mutex.WaitOne(0)) { exit 0 }

function Write-Status([string]$message) {
    Add-Content -LiteralPath $logPath -Value "$(Get-Date -Format o) $message" -Encoding UTF8
}

function Get-SshOptions {
    $ip = Get-NetIPAddress -InterfaceAlias 'WLAN' -AddressFamily IPv4 -ErrorAction SilentlyContinue |
        Where-Object { $_.IPAddress -notlike '169.254.*' } |
        Select-Object -First 1 -ExpandProperty IPAddress
    $sshArgs = @('-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10')
    $scpArgs = @('-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10')
    if ($ip) {
        $sshArgs += @('-b', $ip)
        $scpArgs += @('-o', "BindAddress=$ip")
    }
    return @{ Ssh = $sshArgs; Scp = $scpArgs }
}

function Test-Remote([string]$command, [hashtable]$options) {
    $sshArgs = $options.Ssh
    & ssh @sshArgs $remoteHostName $command *> $null
    return $LASTEXITCODE -eq 0
}

function Copy-Remote([string]$remotePath, [string]$localPath, [hashtable]$options) {
    $source = '{0}:{1}' -f $remoteHostName, $remotePath
    $scpArgs = $options.Scp
    & scp @scpArgs '-p' $source $localPath 2>> $logPath
    if ($LASTEXITCODE -ne 0) { throw "scp failed: $remotePath" }
}

try {
    Write-Status 'Local sync watcher started'
    do {
        try {
            $options = Get-SshOptions
            if (-not (Test-Path -LiteralPath $modelMarker)) {
                if (Test-Remote "test -s '$remoteArchive' && test -s '$remoteArchive.sha256'" $options) {
                    Write-Status 'Model archive ready; downloading'
                    $partialPath = "$archivePath.partial"
                    Copy-Remote $remoteArchive $partialPath $options
                    Copy-Remote "$remoteArchive.sha256" $shaPath $options
                    $expected = ((Get-Content -LiteralPath $shaPath -Raw).Trim() -split '\s+')[0].ToLowerInvariant()
                    $actual = (Get-FileHash -LiteralPath $partialPath -Algorithm SHA256).Hash.ToLowerInvariant()
                    if ($actual -ne $expected) { throw "Archive checksum mismatch: $actual != $expected" }
                    Move-Item -LiteralPath $partialPath -Destination $archivePath -Force
                    & tar -xzf $archivePath -C $projectRoot
                    if ($LASTEXITCODE -ne 0) { throw 'Archive extraction failed' }
                    if (-not (Test-Path -LiteralPath (Join-Path $runDir 'checkpoint-000200\COMPLETE')) -or
                        -not (Test-Path -LiteralPath (Join-Path $runDir 'final\adapter_model.safetensors'))) {
                        throw 'Extracted model is incomplete'
                    }
                    @{ synced_at = (Get-Date -Format o); sha256 = $actual; archive = $archivePath } |
                        ConvertTo-Json | Set-Content -LiteralPath $modelMarker -Encoding UTF8
                    Write-Status "Model, checkpoints and dataset synced: $actual"
                }
            }

            if ((Test-Path -LiteralPath $modelMarker) -and -not (Test-Path -LiteralPath $evalMarker)) {
                $remoteEval = "$remoteRoot/runs/dapo_math_baseline_dev500_4.json"
                if (Test-Remote "test -s '$remoteEval'" $options) {
                    Copy-Remote $remoteEval (Join-Path $projectRoot 'runs\dapo_math_baseline_dev500_4.json') $options
                    Set-Content -LiteralPath $evalMarker -Value (Get-Date -Format o) -Encoding UTF8
                    Write-Status 'DAPO-Math 500-question evaluation synced'
                }
            }

            if ((Test-Path -LiteralPath $modelMarker) -and (Test-Path -LiteralPath $evalMarker)) {
                Write-Status 'Local sync complete'
                Unregister-ScheduledTask -TaskName $taskName -Confirm:$false -ErrorAction SilentlyContinue
                break
            }
        } catch {
            Write-Status "Retrying after error: $_"
        }
        if (-not $Wait) { break }
        Start-Sleep -Seconds 120
    } while ($true)
} finally {
    $mutex.ReleaseMutex()
    $mutex.Dispose()
}
