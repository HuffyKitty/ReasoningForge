$ErrorActionPreference = 'Stop'
$projectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$runDir = Join-Path $projectRoot 'runs'
$logDir = Join-Path $projectRoot 'logs'
$status = Join-Path $logDir 'gsm8k_final_local_sync.log'
$localMarker = Join-Path $runDir 'GSM8K_LOCAL_SYNC_COMPLETE'
$remoteHostName = 'autodl-reasoningforge'
$remoteRoot = '/root/autodl-tmp/ReasoningForge-next'
$remoteManifest = "$remoteRoot/runs/gsm8k_final_results.sha256"
$taskName = 'ReasoningForge-GSM8K-FinalSync'
New-Item -ItemType Directory -Force -Path $runDir, $logDir | Out-Null

$mutex = [System.Threading.Mutex]::new($false, 'Local\ReasoningForgeGSM8KFinalSync')
if (-not $mutex.WaitOne(0)) { exit 0 }

function Write-Status([string]$message) {
    Add-Content -LiteralPath $status -Value "$(Get-Date -Format o) $message" -Encoding UTF8
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

function Invoke-Remote([string]$command, [hashtable]$options) {
    $sshArgs = $options.Ssh
    & ssh @sshArgs $remoteHostName $command *> $null
    return $LASTEXITCODE -eq 0
}

function Copy-Remote([string]$remotePath, [string]$localPath, [hashtable]$options) {
    $scpArgs = $options.Scp
    $source = '{0}:{1}' -f $remoteHostName, $remotePath
    & scp @scpArgs '-p' $source $localPath 2>> $status
    if ($LASTEXITCODE -ne 0) { throw "scp failed: $remotePath" }
}

try {
    if (Test-Path -LiteralPath $localMarker) {
        Unregister-ScheduledTask -TaskName $taskName -Confirm:$false -ErrorAction SilentlyContinue
        exit 0
    }
    $options = Get-SshOptions
    if (-not (Invoke-Remote "test -s '$remoteManifest'" $options)) { exit 0 }

    Write-Status 'Final comparison ready; downloading results'
    $manifestPath = Join-Path $runDir 'gsm8k_final_results.sha256'
    Copy-Remote $remoteManifest $manifestPath $options
    foreach ($line in Get-Content -LiteralPath $manifestPath) {
        if ($line -notmatch '^([0-9a-f]{64})\s+\*?runs/([^/]+)$') {
            throw "Unexpected checksum line: $line"
        }
        $expected = $Matches[1]
        $name = $Matches[2]
        $destination = Join-Path $runDir $name
        if ((Test-Path -LiteralPath $destination) -and
            (Get-FileHash -LiteralPath $destination -Algorithm SHA256).Hash -ieq $expected) {
            continue
        }
        $partial = "$destination.partial"
        Copy-Remote "$remoteRoot/runs/$name" $partial $options
        $actual = (Get-FileHash -LiteralPath $partial -Algorithm SHA256).Hash
        if ($actual -ine $expected) { throw "Checksum mismatch: $name" }
        Move-Item -LiteralPath $partial -Destination $destination -Force
    }
    Copy-Remote '/root/autodl-tmp/logs/gsm8k_finish_after_pass8.status.log' (Join-Path $logDir 'gsm8k_finish_after_pass8.status.log') $options
    if (-not (Invoke-Remote "touch '$remoteRoot/runs/GSM8K_LOCAL_SYNC_COMPLETE'" $options)) {
        throw 'Could not acknowledge local sync to AutoDL'
    }
    Set-Content -LiteralPath $localMarker -Value (Get-Date -Format o) -Encoding UTF8
    Write-Status 'All six result files verified locally; AutoDL shutdown acknowledged'
    Unregister-ScheduledTask -TaskName $taskName -Confirm:$false -ErrorAction SilentlyContinue
} catch {
    Write-Status "Retrying after error: $_"
} finally {
    $mutex.ReleaseMutex()
    $mutex.Dispose()
}
