import contextlib
import os
import socket
import sys
import threading
import time
import webbrowser
from pathlib import Path

from streamlit.web import bootstrap


APP_NAME = "Xper Hemodynamic Viewer"


def resource_path(filename: str) -> Path:
    """Return a file bundled by PyInstaller, or next to this script in dev."""
    base = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))
    bundled_path = base / filename
    if bundled_path.exists():
        return bundled_path
    return base.parent / filename


def find_available_port() -> int:
    with contextlib.closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def wait_for_server(port: int, timeout_s: float = 20.0) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        with contextlib.closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
            sock.settimeout(0.25)
            if sock.connect_ex(("127.0.0.1", port)) == 0:
                return True
        time.sleep(0.25)
    return False


def open_browser_when_ready(port: int) -> None:
    if wait_for_server(port):
        webbrowser.open(f"http://127.0.0.1:{port}")


def configure_streamlit_environment() -> None:
    os.environ.setdefault("STREAMLIT_BROWSER_GATHER_USAGE_STATS", "false")
    os.environ.setdefault("STREAMLIT_SERVER_HEADLESS", "true")
    os.environ.setdefault("STREAMLIT_GLOBAL_DEVELOPMENT_MODE", "false")


def configure_logging() -> None:
    log_dir = Path.home() / "Library" / "Logs" / APP_NAME
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "app.log"
    log_file = open(log_path, "a", buffering=1, encoding="utf-8")
    sys.stdout = log_file
    sys.stderr = log_file
    print(f"\n--- {APP_NAME} started at {time.strftime('%Y-%m-%d %H:%M:%S')} ---")


def main() -> None:
    configure_logging()
    configure_streamlit_environment()

    app_path = resource_path("app.py")
    if not app_path.exists():
        raise FileNotFoundError(f"Bundled Streamlit app not found: {app_path}")

    port = find_available_port()
    os.environ["STREAMLIT_SERVER_ADDRESS"] = "127.0.0.1"
    os.environ["STREAMLIT_SERVER_PORT"] = str(port)
    threading.Thread(target=open_browser_when_ready, args=(port,), daemon=True).start()

    flag_options = {
        "server_address": "127.0.0.1",
        "server_port": port,
        "server_headless": True,
        "browser_gatherUsageStats": False,
        "global_developmentMode": False,
    }
    bootstrap.load_config_options(flag_options)
    bootstrap.run(str(app_path), False, [], flag_options)


if __name__ == "__main__":
    main()
