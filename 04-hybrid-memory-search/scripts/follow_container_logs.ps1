# Persist docker container logs to ./logs/<name>-yyyyMMdd-HHmmss.log (Asia/Shanghai).
# A new file is created when this script starts, and again when the Beijing calendar day changes.
# Usage:
#   powershell -ExecutionPolicy Bypass -File scripts\follow_container_logs.ps1
#   powershell -ExecutionPolicy Bypass -File scripts\follow_container_logs.ps1 -Containers hybrid-memory,timem-adapter

param(
    [string[]]$Containers = @("hybrid-memory")
)

$ErrorActionPreference = "Continue"
$Root = Split-Path -Parent $PSScriptRoot
$LogDir = Join-Path $Root "logs"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

function Get-BeijingNow {
    try {
        return [System.TimeZoneInfo]::ConvertTimeBySystemTimeZoneId(
            [DateTime]::UtcNow,
            "China Standard Time"
        )
    } catch {
        # Fallback: UTC+8
        return [DateTime]::UtcNow.AddHours(8)
    }
}

function Get-LogPath {
    param(
        [string]$Name,
        [datetime]$When,
        [string]$Suffix = ""
    )
    $stamp = $When.ToString("yyyyMMdd-HHmmss")
    $file = if ($Suffix) { "{0}-{1}-{2}.log" -f $Name, $stamp, $Suffix } else { "{0}-{1}.log" -f $Name, $stamp }
    return Join-Path $LogDir $file
}

# One-time snapshot of current buffer into a Beijing-time named file
$bj = Get-BeijingNow
foreach ($Name in $Containers) {
    $snap = Get-LogPath -Name $Name -When $bj -Suffix "snapshot"
    $ts = $bj.ToString("yyyy-MM-ddTHH:mm:ss+08:00")
    Add-Content -Path $snap -Encoding utf8 -Value "[$ts] snapshot docker logs --timestamps --tail 5000 $Name"
    docker logs --timestamps --tail 5000 $Name 1>> $snap 2>> $snap
    Write-Host "snapshot -> $snap"
}

$jobs = @()
foreach ($Name in $Containers) {
    $jobs += Start-Job -Name "log-$Name" -ScriptBlock {
        param($Name, $LogDir)

        function Get-BeijingNow {
            try {
                return [System.TimeZoneInfo]::ConvertTimeBySystemTimeZoneId(
                    [DateTime]::UtcNow,
                    "China Standard Time"
                )
            } catch {
                return [DateTime]::UtcNow.AddHours(8)
            }
        }

        function Open-DayFile {
            param([string]$Name, [string]$LogDir, [datetime]$When)
            $day = $When.ToString("yyyyMMdd")
            $stamp = $When.ToString("yyyyMMdd-HHmmss")
            $path = Join-Path $LogDir ("{0}-{1}.log" -f $Name, $stamp)
            # Prefer one file per Beijing day: reuse if same-day file already exists
            $existing = Get-ChildItem -Path $LogDir -Filter ("{0}-{1}-*.log" -f $Name, $day) -ErrorAction SilentlyContinue |
                Where-Object { $_.Name -notmatch "snapshot" } |
                Sort-Object LastWriteTime |
                Select-Object -Last 1
            if ($existing) { $path = $existing.FullName }
            $ts = $When.ToString("yyyy-MM-ddTHH:mm:ss+08:00")
            Add-Content -Path $path -Encoding utf8 -Value "[$ts] ---- follow docker logs -f --timestamps $Name ----"
            return @{ Path = $path; Day = $day }
        }

        $state = Open-DayFile -Name $Name -LogDir $LogDir -When (Get-BeijingNow)
        Write-Output ("following {0} -> {1}" -f $Name, $state.Path)

        while ($true) {
            $bj = Get-BeijingNow
            $today = $bj.ToString("yyyyMMdd")
            if ($today -ne $state.Day) {
                $state = Open-DayFile -Name $Name -LogDir $LogDir -When $bj
            }

            $running = docker inspect -f "{{.State.Running}}" $Name 2>$null
            if ($running -ne "true") {
                $ts = $bj.ToString("yyyy-MM-ddTHH:mm:ss+08:00")
                Add-Content -Path $state.Path -Encoding utf8 -Value "[$ts] container $Name not running; retry in 5s"
                Start-Sleep -Seconds 5
                continue
            }

            # Stream until container stops/recreates; then reconnect.
            # Use cmd redirection so stdout+stderr append to the Beijing-dated file.
            $path = $state.Path
            cmd /c "docker logs -f --timestamps --tail 0 $Name >> `"$path`" 2>&1"

            $bj = Get-BeijingNow
            $ts = $bj.ToString("yyyy-MM-ddTHH:mm:ss+08:00")
            # If day rolled during follow, switch file before next reconnect note
            if ($bj.ToString("yyyyMMdd") -ne $state.Day) {
                $state = Open-DayFile -Name $Name -LogDir $LogDir -When $bj
            }
            Add-Content -Path $state.Path -Encoding utf8 -Value "[$ts] ---- docker logs exited; reconnect in 2s ----"
            Start-Sleep -Seconds 2
        }
    } -ArgumentList $Name, $LogDir
}

Write-Host "log dir: $LogDir (Beijing time filenames: <container>-yyyyMMdd-HHmmss.log)"
Write-Host "press Ctrl+C to stop following"
try {
    while ($true) {
        Start-Sleep -Seconds 30
        foreach ($j in $jobs) {
            Receive-Job $j -ErrorAction SilentlyContinue | ForEach-Object { Write-Host $_ }
        }
    }
} finally {
    $jobs | Stop-Job -ErrorAction SilentlyContinue
    $jobs | Remove-Job -Force -ErrorAction SilentlyContinue
}
