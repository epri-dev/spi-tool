import click
import panel as pn

from .ui import create_app
from .version import __version__


@click.group()
@click.version_option(__version__)
def cli():
    """spi-tool CLI"""
    pass


@cli.command()
@click.option(
    "--show/--no-show",
    help="Launch a web-browser showing the app",
    default=True,
)
@click.option(
    "--port",
    help="Port to run the app on",
    default=None,
)
def dashboard(show=True, port=None):
    """
    Run the dashboard
    """
    app = create_app()
    if port is not None:
        pn.serve(app, show=show, port=port)
    else:
        pn.serve(app, show=show)


if __name__ == "__main__":
    cli()
