"""Download pre-trained ResMatching checkpoints.

Usage:
    uv run python scripts/download_models.py               # download all pre-trained models
    uv run python scripts/download_models.py --checkpoint-dir /my/path
    uv run python scripts/download_models.py --subset ccp --subset er
"""

from enum import Enum
from pathlib import Path
from typing import Annotated, Optional

import pooch
import typer

BASE_URL = "https://download.fht.org/jug/resmatching/checkpoints/"

DESCRIPTIONS = {
    "ccp": "Clathrin-Coated Pits",
    "er": "Endoplasmic Reticulum",
    "factin": "F-actin",
    "mt": "Microtubules",
    "mt_noisy": "Microtubules (noisy input)",
}

MODELS: dict[str, dict[str, str]] = {
    "ccp": {
        "best_model.pth": "5e0d06074b3fad83000bf18f8ed18d4fa286aa36c31acaa41bc96df356045d85"
    },
    "er": {
        "best_model.pth": "dbd629fef8460a340940dde4b74881874124d121cebaea997a350ad0e101fb16"
    },
    "factin": {
        "best_model.pth": "34a59b21bdda3039a24a2c4d3b64985fb128903b993bdbf1f62c851f114663ed"
    },
    "mt": {
        "best_model.pth": "a5d2269738007c7cd966ab9d4f26b0ea2ed93839f3315dbb89bc0f079ba5de7e"
    },
    "mt_noisy": {
        "best_model.pth": "64083cb0fefb9f6875066d42ad6df386c6fa0c9e0cf6321a2c1f2265a8a68d10"
    },
}

Subset = Enum("Subset", {k: k for k in MODELS})

app = typer.Typer()


def _download_subset(key: str, checkpoint_dir: Path) -> None:
    dest = checkpoint_dir / key
    dest.mkdir(parents=True, exist_ok=True)
    typer.echo(f"Downloading {DESCRIPTIONS[key]} checkpoints -> {dest}/")
    for filename, known_hash in MODELS[key].items():
        pooch.retrieve(
            url=BASE_URL + f"{key}/{filename}",
            known_hash=known_hash,
            fname=filename,
            path=dest,
            progressbar=True,
        )
        typer.echo(f"  {filename}")


@app.command()
def main(
    checkpoint_dir: Annotated[
        Path, typer.Option(help="Directory to download checkpoints into.")
    ] = Path("checkpoints"),
    subset: Annotated[
        Optional[list[Subset]],
        typer.Option(
            help="Subset(s) to download. Repeat to select multiple. Default: all."
        ),
    ] = None,
):
    keys = [s.value for s in subset] if subset else list(MODELS)

    for key in keys:
        _download_subset(key, checkpoint_dir)

    typer.echo("\nAll done. Checkpoint layout:")
    for key in keys:
        typer.echo(f"  {checkpoint_dir / key}/")
        for filename in MODELS[key]:
            typer.echo(f"    {filename}")


if __name__ == "__main__":
    app()
