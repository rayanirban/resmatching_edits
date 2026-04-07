"""Download pre-trained ResMatching checkpoints.

Usage:
    uv run python download_models.py               # download all subsets
    uv run python download_models.py --checkpoint-dir /my/path
    uv run python download_models.py --subset ccp --subset er
"""

from enum import Enum
from pathlib import Path
from typing import Annotated, Optional

import pooch
import typer

BASE_URL = "https://download.fht.org/jug/Anirban/ISBI2026/checkpoints/"

DESCRIPTIONS = {
    "ccp":      "Clathrin-Coated Pits",
    "er":       "Endoplasmic Reticulum",
    "factin":   "F-actin",
    "mt":       "Microtubules",
    "mt_noisy": "Microtubules (noisy input)",
}

# Each entry: subset key -> {filename: sha256 or None}
# Update hashes once files are public.
MODELS: dict[str, dict[str, str | None]] = {
    "ccp":      {"best_model.pth": None, "last_model.pth": None},
    "er":       {"best_model.pth": None, "last_model.pth": None},
    "factin":   {"best_model.pth": None, "last_model.pth": None},
    "mt":       {"best_model.pth": None, "last_model.pth": None},
    "mt_noisy": {"best_model.pth": None, "last_model.pth": None},
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
    checkpoint_dir: Annotated[Path, typer.Option(help="Directory to download checkpoints into.")] = Path("checkpoints"),
    subset: Annotated[Optional[list[Subset]], typer.Option(help="Subset(s) to download. Repeat to select multiple. Default: all.")] = None,
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
