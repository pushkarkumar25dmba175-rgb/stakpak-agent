<#
.SYNOPSIS
    Register PersonalOS Agent to start with Windows.

.DESCRIPTION
    Creates a Scheduled Task that runs `agent schedule serve` at logon, as the
    current user. It does NOT run with highest privileges: the agent has no
    business being elevated, and a scheduled task that is would be a standing
    privilege-escalation path.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File register-task.ps1
    Unregister-ScheduledTask -TaskName PersonalOSAgent -Confirm:$false
#>

$ErrorActionPreference = 'Stop'

$agent = (Get-Command agent -ErrorAction SilentlyContinue).Source
if (-not $agent) {
    throw "Could not find 'agent' on PATH. Install PersonalOS, or edit this script to use the full path."
}

$action = New-ScheduledTaskAction -Execute $agent -Argument 'schedule serve'
$trigger = New-ScheduledTaskTrigger -AtLogOn -User $env:USERNAME
$principal = New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType Interactive -RunLevel Limited
$settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -StartWhenAvailable `
    -RestartCount 3 `
    -RestartInterval (New-TimeSpan -Minutes 1)

Register-ScheduledTask `
    -TaskName 'PersonalOSAgent' `
    -Description 'PersonalOS Agent scheduler service (runs as the logged-in user).' `
    -Action $action `
    -Trigger $trigger `
    -Principal $principal `
    -Settings $settings `
    -Force | Out-Null

Write-Host "Registered. Check it with: agent status"
Write-Host "Remove it with:  Unregister-ScheduledTask -TaskName PersonalOSAgent -Confirm:`$false"
