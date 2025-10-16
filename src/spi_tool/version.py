import subprocess
import shlex
from importlib import metadata
import os
import sys
import datetime


DIR = os.path.dirname(os.path.realpath(__file__))

VERSION = metadata.version("spi_tool")


def cmd(cmd, kind="") -> str:
    """
    Get git subprocess output
    """
    output = "unknown"
    try:
        output = (
            subprocess.check_output(shlex.split(cmd), cwd=DIR, stderr=subprocess.STDOUT)
            .decode()
            .strip()
        )
    except Exception as _:
        ...
    return f"{kind}{output}"


def last_commit_id() -> str:
    return cmd("git describe --always --dirty")


def branch() -> str:
    return cmd("git rev-parse --abbrev-ref HEAD")


def get_git_version() -> str:
    return f"{last_commit_id()}-{branch()}"


__version__ = f"{VERSION}-{get_git_version()}"


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "full-version":
        current_date = datetime.datetime.now().strftime("%Y-%m-%d")
        print(f"v{__version__}-{current_date}")
    else:
        print("v" + __version__)
