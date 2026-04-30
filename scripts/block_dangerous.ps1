param([string]$Command = $env:CLAUDE_TOOL_INPUT)
if ($Command -match "rm -rf.*(models[/\\]|data[/\\]processed|data[/\\]raw|mlruns|reports[/\\])") {
    Write-Error "BLOCKED: destructive command on protected folder"
    exit 1
}
if ($Command -match "dvc destroy|dvc gc.*--force") {
    Write-Error "BLOCKED: destructive DVC command"
    exit 1
}
exit 0
