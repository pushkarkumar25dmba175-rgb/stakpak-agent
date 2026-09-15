# Starting PersonalOS with your computer

These templates start the **agent service only** — the scheduler and, if you
enabled it, the folder watcher. Starting the service does not start doing
things: scheduled jobs run on their schedule, everything else waits for you.

Nothing here runs as root, and nothing here needs to.

| Platform | File | Install |
|---|---|---|
| Linux | `systemd/personalos-agent.service` | `systemctl --user enable --now personalos-agent` |
| macOS | `launchd/com.personalos.agent.plist` | `launchctl load ~/Library/LaunchAgents/com.personalos.agent.plist` |
| Windows | `windows/register-task.ps1` | `powershell -ExecutionPolicy Bypass -File register-task.ps1` |

Each template assumes `agent` is on your `PATH`. If you installed into a
virtualenv, use the absolute path to that environment's `agent` instead.

To check it is working: `agent status`, then `agent schedule list`.
To stop it: `systemctl --user disable --now personalos-agent`, `launchctl
unload …`, or `Unregister-ScheduledTask -TaskName PersonalOSAgent`.
