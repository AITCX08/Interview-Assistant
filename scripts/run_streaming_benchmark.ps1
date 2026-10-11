param(
    [Parameter(Mandatory=$true)][string]$Audio,
    [Parameter(Mandatory=$true)][int]$InputDevice,
    [Parameter(Mandatory=$true)][int]$OutputDevice,
    [switch]$RealAI,
    [string]$Environment = 'C:\Coding\Anaconda\envs\Interview-Assistant'
)
$ErrorActionPreference = 'Stop'
$projectPath = Split-Path $PSScriptRoot -Parent
$benchmarkAudio = (Resolve-Path -LiteralPath $Audio).Path
$benchmarkRunId = 'streaming-' + (Get-Date -Format 'yyyyMMdd-HHmmss') + '-' + [guid]::NewGuid().ToString('N').Substring(0, 8)
$benchmarkRunPath = Join-Path $projectPath ('.local\' + $benchmarkRunId)
if (Test-Path -LiteralPath $benchmarkRunPath) { throw 'RunId already exists' }
New-Item -ItemType Directory -Path $benchmarkRunPath | Out-Null
$benchmarkScript = Join-Path $PSScriptRoot 'benchmark_streaming.py'
$benchmarkArguments = @('run', '--no-capture-output', '-p', ('"' + $Environment + '"'),
    'python', '-u', ('"' + $benchmarkScript + '"'), '--run-dir', ('"' + $benchmarkRunPath + '"'),
    '--audio', ('"' + $benchmarkAudio + '"'), '--input-device', $InputDevice, '--output-device', $OutputDevice)
if ($RealAI) { $benchmarkArguments += '--real-ai' }
$benchmarkProcess = Start-Process -FilePath (Get-Command conda.exe).Source -ArgumentList $benchmarkArguments -WorkingDirectory $projectPath -WindowStyle Hidden -Environment @{TEMP=$benchmarkRunPath;TMP=$benchmarkRunPath;PYTHONIOENCODING='utf-8'} -RedirectStandardOutput (Join-Path $benchmarkRunPath 'launcher.stdout.log') -RedirectStandardError (Join-Path $benchmarkRunPath 'launcher.stderr.log') -PassThru
[System.IO.File]::WriteAllText((Join-Path $benchmarkRunPath 'launcher-pid.txt'), [string]$benchmarkProcess.Id)
[System.IO.File]::WriteAllText((Join-Path $projectPath '.local\latest-streaming.txt'), $benchmarkRunPath)
[pscustomobject]@{RunId=$benchmarkRunId;LauncherPid=$benchmarkProcess.Id;Directory=$benchmarkRunPath} | ConvertTo-Json -Compress
