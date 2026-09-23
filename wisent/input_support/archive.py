"""Safe extraction helpers for immutable model and dataset archives."""
from pathlib import Path, PurePosixPath
import shutil
import tarfile

from wisent.stado import StadoError

_COPY_CHUNK_BYTES = int("1048576")


def empty_destination(destination: Path) -> None:
    if destination.exists() and (not destination.is_dir() or any(destination.iterdir())):
        raise StadoError(f"input destination must be an empty directory: {destination}")
    destination.mkdir(parents=True, exist_ok=True)


def extract_archive(archive_path: Path, destination: Path) -> None:
    """Extract regular files and directories without traversal or links."""
    empty_destination(destination)
    root = destination.resolve()
    try:
        archive = tarfile.open(archive_path, mode="r:gz")
    except tarfile.TarError as exc:
        raise StadoError(f"invalid gzip tar input: {archive_path}") from exc
    with archive:
        for member in archive.getmembers():
            relative = PurePosixPath(member.name)
            if relative.is_absolute() or ".." in relative.parts:
                raise StadoError(f"unsafe archive member: {member.name}")
            parts = tuple(part for part in relative.parts if part not in {"", "."})
            if not parts:
                if member.isdir():
                    continue
                raise StadoError(f"unsafe archive member: {member.name}")
            target = destination.joinpath(*parts)
            resolved = target.resolve()
            if root != resolved and root not in resolved.parents:
                raise StadoError(f"archive member escapes destination: {member.name}")
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            if not member.isfile():
                raise StadoError(f"archive member is not a regular file: {member.name}")
            target.parent.mkdir(parents=True, exist_ok=True)
            source = archive.extractfile(member)
            if source is None:
                raise StadoError(f"archive member has no content: {member.name}")
            with source, target.open("xb") as output:
                shutil.copyfileobj(source, output, length=_COPY_CHUNK_BYTES)


def model_root(extracted: Path) -> Path:
    if (extracted / "config.json").is_file():
        return extracted
    candidates = [
        child
        for child in extracted.iterdir()
        if child.is_dir()
        and not child.is_symlink()
        and (child / "config.json").is_file()
    ]
    if len(candidates) != int("1"):
        raise StadoError("model archive must contain exactly one model config.json root")
    return next(iter(candidates))
