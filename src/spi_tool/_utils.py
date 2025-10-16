import dataclasses

import click
import os
import sys

from . import version

from importlib.resources import files


def get_resources_path():
    if hasattr(sys, "_MEIPASS"):
        return os.path.join(sys._MEIPASS, "resources")
    else:
        return str(files("spi_tool") / "resources")


DATA_FOLDER_PATH = os.path.join(get_resources_path(), "data")
IMAGE_FOLDER_PATH = os.path.join(get_resources_path(), "images")


def write_version_file():
    with open(os.path.join(get_resources_path(), "version.txt"), "w") as f:
        f.write(version.__version__)


def read_version_file():
    if not os.path.exists(os.path.join(get_resources_path(), "version.txt")):
        return "v0.0.0"
    with open(os.path.join(get_resources_path(), "version.txt")) as f:
        return f.read()


class ColorScheme:
    def __init__(self, fg: str, bg: str):
        self.fg = fg
        self.bg = bg


@dataclasses.dataclass(frozen=True)
class Colors:
    print: ColorScheme = dataclasses.field(
        default_factory=lambda: ColorScheme(fg="bright_white", bg="bright_black")
    )
    debug: ColorScheme = dataclasses.field(
        default_factory=lambda: ColorScheme(fg="magenta", bg="bright_black")
    )
    log: ColorScheme = dataclasses.field(
        default_factory=lambda: ColorScheme(fg="green", bg="bright_black")
    )
    tip: ColorScheme = dataclasses.field(
        default_factory=lambda: ColorScheme(fg="cyan", bg="bright_black")
    )
    info: ColorScheme = dataclasses.field(
        default_factory=lambda: ColorScheme(fg="blue", bg="bright_white")
    )
    warn: ColorScheme = dataclasses.field(
        default_factory=lambda: ColorScheme(fg="bright_yellow", bg="bright_black")
    )
    error: ColorScheme = dataclasses.field(
        default_factory=lambda: ColorScheme(fg="red", bg="bright_white")
    )


# This singleton variable can be used to access colors from any file
COLORS = Colors()


def echo(
    *messages,
    sep: str = " ",
    prefix: str = "PRINT",
    fg: str = COLORS.print.fg,
    bg: str = COLORS.print.bg,
    **kwargs,
):
    """
    Prints a formatted message to the console with customizable prefix and color styling.

    Args:
        *messages: Variable length argument list of messages to be printed.

    Keyword Arguments:
        sep (str): Separator between messages. Default is a single space.
        prefix (str): Prefix text to prepend to the message. Default is "PRINT".
        fg (str): Foreground color for the text. Default is the color defined in COLORS.print["fg"].
        bg (str): Background color for the text. Default is the color defined in COLORS.print["bg"].
        **kwargs: Additional keyword arguments to pass to `click.echo`. This can include options like `blink` and `underline`.
        blink (bool): Whether the text should blink. Default is False.
        underline (bool): Whether the text should be underlined. Default is False.

    Example:

    >>> echo("Hello", "World", sep=", ", prefix="INFO", fg="yellow", bg="blue", blink=True)
    """
    blink = kwargs.pop("blink", False)
    underline = kwargs.pop("underline", False)

    message = sep.join([f"{m}" for m in messages])

    styled_spacer = click.style(" ", bg=fg)
    styled_prefix = click.style(
        prefix, bg=fg, fg=bg, blink=blink, bold=True, underline=underline
    )
    styled_message = click.style(f": {message}")

    click.echo(
        f"{styled_spacer}{styled_prefix}{styled_spacer}{styled_message}",
        **kwargs,
    )


def debug(*args, **kwargs):
    """
    Prints a message DEBUG message
    """
    echo(*args, prefix="DEBUG", **COLORS.debug, **kwargs)


def log(*args, **kwargs):
    """
    Prints a message LOG message
    """
    echo(*args, prefix="LOG", **COLORS.log, **kwargs)


def tip(*args, **kwargs):
    """
    Prints a TIP message.
    """
    echo(*args, prefix="TIP", **COLORS.tip, **kwargs)


def info(*args, **kwargs):
    """
    Prints an INFO message.
    """
    echo(*args, prefix="INFO", **COLORS.info, **kwargs)


def warn(*args, **kwargs):
    """
    Prints a WARN message.
    """
    echo(*args, prefix="WARN", **COLORS.warn, **kwargs)


def error(*args, **kwargs):
    """
    Prints an ERR message.
    """
    echo(*args, prefix="ERROR", **COLORS.error, err=True, **kwargs)


divider_stylesheet = """
.bk-clearfix hr {
    border:none;
    border-top: 1px solid var(--design-secondary-color, var(--panel-secondary-color));
}
:host(.bk-above) .bk-header .bk-tab {font-size: 1.25em;}
.bk-active {
    color: white !important;
    background-color: var(--pn-tab-active-color) !important;
    font-weight: bold;
}
"""
