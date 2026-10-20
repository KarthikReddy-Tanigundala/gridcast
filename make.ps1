<#
    GridCast task runner for Windows / PowerShell (twin of the Makefile).
    Usage:  .\make.ps1 <target>          e.g.  .\make.ps1 backtest
    Targets:
      setup      venv + pinned Python deps (editable install)
      setup-dl   optional deep-learning extras (neuralforecast + CPU torch)
      fetch      AEMO actuals + Open-Meteo weather -> data/raw (incremental)
      fetch-aemo AEMO PREDISPATCH forecasts (nemseer) -> data/raw
      build      join actuals + weather -> data/processed/vic1_30min.parquet
      backtest   walk-forward backtest -> results/leaderboard.md
      benchmark  GridCast vs AEMO PREDISPATCH -> results/aemo_benchmark_results.json
      spikes     P(RRP > $300) classifier backtest
      dl         deep-learning comparison (needs setup-dl)
      train      fit final models -> models/*.joblib
      predict    next-day forecast -> data/forecasts/
      refresh    daily job: fetch -> build -> predict
      app        run the Streamlit dashboard
      test       pytest
      lint       ruff check + format check
      all        fetch -> build -> backtest -> benchmark -> train -> predict
#>
param(
    [Parameter(Position = 0)]
    [ValidateSet('setup', 'setup-dl', 'fetch', 'fetch-aemo', 'build', 'backtest', 'benchmark', 'spikes', 'dl',
                 'train', 'predict', 'refresh', 'app', 'test', 'lint', 'all')]
    [string]$Target = 'all'
)

$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot

$Py = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path $Py)) { $Py = 'python' }

function Invoke-Checked([string]$Exe, [string[]]$Arguments) {
    # Windows PowerShell 5.1 turns a native tool's stderr (Python logging) into terminating errors
    # under 'Stop', so relax it for the call and rely on the exit code instead.
    $prev = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try { & $Exe @Arguments 2>&1 | ForEach-Object { "$_" } }
    finally { $ErrorActionPreference = $prev }
    if ($LASTEXITCODE -ne 0) { throw "$Exe $($Arguments -join ' ') failed with exit code $LASTEXITCODE" }
}

function Invoke-Module([string]$Module) { Invoke-Checked $Py @('-m', $Module) }

switch ($Target) {
    'setup'      { py -3.12 -m venv .venv; Invoke-Checked $Py @('-m', 'pip', 'install', '-r', 'requirements.txt'); Invoke-Checked $Py @('-m', 'pip', 'install', '-e', '.') }
    'setup-dl'   { Invoke-Checked $Py @('-m', 'pip', 'install', '-r', 'requirements-dl.txt') }
    'fetch'      { Invoke-Module 'gridcast.data.fetch_actuals'; Invoke-Module 'gridcast.data.fetch_weather' }
    'fetch-aemo' { Invoke-Module 'gridcast.data.fetch_aemo_forecasts' }
    'build'      { Invoke-Module 'gridcast.data.build_dataset' }
    'backtest'   { Invoke-Module 'gridcast.evaluation.backtest' }
    'benchmark'  { Invoke-Module 'gridcast.evaluation.aemo_benchmark' }
    'spikes'     { Invoke-Module 'gridcast.evaluation.spike_classification' }
    'dl'         { Invoke-Module 'gridcast.models.deep_learning' }
    'train'      { Invoke-Module 'gridcast.train' }
    'predict'    { Invoke-Module 'gridcast.predict' }
    'refresh'    { foreach ($t in 'fetch', 'build', 'predict') { & $PSCommandPath $t } }
    'app'        { Invoke-Checked $Py @('-m', 'streamlit', 'run', 'app/streamlit_app.py') }
    'test'       { Invoke-Checked $Py @('-m', 'pytest') }
    'lint'       { Invoke-Checked $Py @('-m', 'ruff', 'check', '.'); Invoke-Checked $Py @('-m', 'ruff', 'format', '--check', '.') }
    'all'        { foreach ($t in 'fetch', 'build', 'backtest', 'benchmark', 'train', 'predict') { & $PSCommandPath $t } }
}
