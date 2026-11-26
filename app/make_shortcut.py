"""在桌面创建「本地配音工作室」快捷方式，指向本文件夹里的启动脚本。"""
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paths  # noqa: E402

bat = paths.ROOT / "启动 FengStudio.bat"
ps = (
    "$d=[Environment]::GetFolderPath('Desktop');"
    "$s=(New-Object -ComObject WScript.Shell).CreateShortcut((Join-Path $d 'FengStudio.lnk'));"
    f"$s.TargetPath='{bat}';$s.WorkingDirectory='{paths.ROOT}';"
    "$s.IconLocation=\"$env:SystemRoot\\System32\\SHELL32.dll,168\";$s.Save();"
    "Write-Output (Join-Path $d 'FengStudio.lnk')"
)
r = subprocess.run(["powershell", "-NoProfile", "-Command", ps], capture_output=True, text=True, encoding="utf-8",
                   errors="replace")
print("已创建桌面快捷方式：" + r.stdout.strip() if r.returncode == 0 else "创建失败：" + r.stderr[-300:])
