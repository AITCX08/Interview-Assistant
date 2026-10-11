param(
    [string]$Pattern = 'test_*.py',
    [string]$Environment = 'C:\Coding\Anaconda\envs\Interview-Assistant'
)
$ErrorActionPreference = 'Stop'
$projectPath = Split-Path $PSScriptRoot -Parent
$checkRunId = 'checks-' + (Get-Date -Format 'yyyyMMdd-HHmmss') + '-' + [guid]::NewGuid().ToString('N').Substring(0, 8)
$checkRunPath = Join-Path $projectPath ('.local\' + $checkRunId)
if (Test-Path -LiteralPath $checkRunPath) { throw 'RunId already exists' }
New-Item -ItemType Directory -Path $checkRunPath | Out-Null
$condaCommand = (Get-Command conda.exe).Source
$checkScript = Join-Path $PSScriptRoot 'check_runner.py'
$env:PYTHONIOENCODING = 'utf-8'
$checkArguments = @('run', '--no-capture-output', '-p', ('"' + $Environment + '"'),
    'python', '-u', ('"' + $checkScript + '"'), '--run-dir', ('"' + $checkRunPath + '"'), '--pattern', $Pattern)
$checkProcess = Start-Process -FilePath $condaCommand -ArgumentList $checkArguments -WorkingDirectory $projectPath -WindowStyle Hidden -RedirectStandardOutput (Join-Path $checkRunPath 'launcher.stdout.log') -RedirectStandardError (Join-Path $checkRunPath 'launcher.stderr.log') -PassThru
[System.IO.File]::WriteAllText((Join-Path $checkRunPath 'launcher-pid.txt'), [string]$checkProcess.Id)
[System.IO.File]::WriteAllText((Join-Path $projectPath '.local\latest-checks.txt'), $checkRunPath)
[pscustomobject]@{RunId=$checkRunId;LauncherPid=$checkProcess.Id;Directory=$checkRunPath} | ConvertTo-Json -Compress
