print("Launching SPI-Tool dashboard...")

import spi_tool  # noqa

spi_tool.cli.create_app().servable()

if __name__ == "__main__":
    # Create a Click context
    with spi_tool.cli.cli.make_context("cli", ["dashboard", "--show"]) as ctx:
        # Call the dashboard subcommand
        spi_tool.cli.cli.invoke(ctx)
