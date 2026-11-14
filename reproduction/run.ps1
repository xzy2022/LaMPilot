param(
    [ValidateSet('offline', 'api')][string]$Mode = 'offline',
    [ValidateSet('demo', 'table3', 'all')][string]$Dataset = 'table3',
    [int]$Limit = 98,
    [string]$Model = 'deepseek-flash',
    [Parameter(Mandatory=$true)][ValidateNotNullOrEmpty()][string]$Output,
    [switch]$ZeroShot
)

$ErrorActionPreference = 'Stop'
$outputPath = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($Output)
Push-Location (Split-Path $PSScriptRoot -Parent)
try {
    $runArgs = @('run', '--no-capture-output', '-n', 'lampilot_py311', 'python', '-m', 'reproduction.run_small', '--mode', $Mode, '--dataset', $Dataset, '--limit', "$Limit", '--model', $Model)
    $runArgs += @('--output', $outputPath)
    if ($ZeroShot) { $runArgs += '--zero-shot' }
    & conda @runArgs
    $runExitCode = $LASTEXITCODE
}
finally {
    Pop-Location
}
exit $runExitCode
