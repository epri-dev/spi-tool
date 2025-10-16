@default:
  just --list

version := `uv run python src/spi_tool/version.py full-version`
datetime := `date +'%Y-%m-%dT%H-%M-%S'`
name := 'spi-tool-dashboard-' + version + '-' + os()

sync:
  uv sync --all-extras --dev --upgrade

generate-version:
  @echo "Generating version.txt file"
  @uv run python -c "import spi_tool; spi_tool._utils.write_version_file()"

build: generate-version
  echo "Building for {{os()}}"
  rm -rf dist
  rm -rf build
  uv run pyinstaller -n '{{name}}' --onefile --noconfirm --clean --add-data "src/spi_tool/resources:resources" main.py
  rm {{name}}.spec

changelog:
  git cliff -o CHANGELOG.md
  npx prettier --write CHANGELOG.md
