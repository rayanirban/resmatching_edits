"""Download BioSR datasets used in ResMatching.

Usage:
    uv run python scripts/download_data.py               # download all subsets
    uv run python scripts/download_data.py --data-dir /my/path
    uv run python scripts/download_data.py --subset ccp --subset er
"""

import zipfile
from enum import Enum
from pathlib import Path
from typing import Annotated, Optional

import pooch
import typer

BASE_URL = "https://download.fht.org/jug/resmatching/data/"


DATASETS = {
    "ccp": (
        "ccp.zip",
        "cf6d2a6838a3f494e333ea28f533a876335d40b4a65ff863150271b9577d9dda",
    ),
    "er": (
        "er.zip",
        "f71724834db7fdff95a3334f2dc1fcfff7aec43e2b2daee14f6d29fa1c9a1b0a",
    ),
    "factin": (
        "factin.zip",
        "1da19967da54df576d4533d458272d3e8f684d13a0840f7be633f52ee95a1965",
    ),
    "mt": (
        "mt.zip",
        "b943e8212bc83d3e8349155421bf96fe346dd6533c5aefc6679dc50a25976f9b",
    ),
    "mt_noisy": (
        "mt_noisy.zip",
        "6208ba16ac6cc79308a3c6eddab47a628bb2b75d34012fc0fad2d786ed63e846",
    ),
}

DESCRIPTIONS = {
    "ccp": "Clathrin-Coated Pits",
    "er": "Endoplasmic Reticulum",
    "factin": "F-actin",
    "mt": "Microtubules",
    "mt_noisy": "Microtubules (noisy input)",
}

Subset = Enum("Subset", {k: k for k in DATASETS})

app = typer.Typer()


def _download_subset(key: str, data_dir: Path) -> None:
    filename, known_hash = DATASETS[key]
    typer.echo(f"Downloading {DESCRIPTIONS[key]} ({filename}) ...")

    path = pooch.retrieve(
        url=BASE_URL + filename,
        known_hash=known_hash,
        fname=filename,
        path=data_dir,
        progressbar=True,
    )

    typer.echo(f"  Extracting to {data_dir} ...")
    with zipfile.ZipFile(path, "r") as zf:
        zf.extractall(data_dir)
    Path(path).unlink()
    typer.echo(f"  Done -> {data_dir / filename.replace('.zip', '')}")


@app.command()
def main(
    data_dir: Annotated[
        Path, typer.Option(help="Directory to download data into.")
    ] = Path("data"),
    subset: Annotated[
        Optional[list[Subset]],
        typer.Option(
            help="Subset(s) to download. Repeat to select multiple. Default: all."
        ),
    ] = None,
):
    keys = [s.value for s in subset] if subset else list(DATASETS)
    data_dir.mkdir(parents=True, exist_ok=True)

    for key in keys:
        _download_subset(key, data_dir)

    typer.echo("\nAll done. Data layout:")
    for key in keys:
        typer.echo(f"  {data_dir / key}/")


if __name__ == "__main__":
    app()
