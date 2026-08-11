"""Find an ffmpeg to run.

A system ffmpeg is always preferred: it is the one the distro built against
the local audio stack, and it costs nothing to use. When there is none, we
fall back to the static build shipped by the `static-ffmpeg` dependency, which
is fetched once into the sttop data directory. That fallback is what makes
`uvx sttop` work on a machine where nobody has installed ffmpeg - the whole
point being that the first run should record audio, not print a package name.

The fetch is deliberately lazy and only ever happens on the miss: importing
this module never touches the network.
"""

from __future__ import annotations

import shutil
from functools import cache


class FFmpegMissing(RuntimeError):
    pass


#: Printed when there is neither a system ffmpeg nor a usable static one.
_HELP = (
    "no ffmpeg available, and the bundled build could not be fetched ({reason}).\n"
    "Install one with your package manager - "
    "`sudo apt install ffmpeg`, `brew install ffmpeg`, `sudo pacman -S ffmpeg`."
)


@cache
def ffmpeg_bin() -> str:
    """Absolute path (or bare name) of an ffmpeg that can be executed."""
    system = shutil.which("ffmpeg")
    if system:
        return system
    return _fetch_static()


def _fetch_static() -> str:
    from ..config import DATA_DIR

    try:
        from static_ffmpeg import run as static_run
    except ImportError as exc:  # dependency stripped from the install
        raise FFmpegMissing(
            _HELP.format(reason=f"static-ffmpeg not installed: {exc}")
        ) from exc

    # static-ffmpeg extracts one level *above* the directory it is given,
    # because its zips are rooted at a platform folder ("linux", "darwin").
    # So the path handed over has to end in that same platform name, or the
    # binaries land beside the directory it then goes looking in.
    import os

    target = DATA_DIR / "ffmpeg" / os.path.basename(static_run.get_platform_dir())
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        ffmpeg, _ffprobe = static_run.get_or_fetch_platform_executables_else_raise(
            download_dir=str(target)
        )
    except Exception as exc:  # no network, unsupported platform, corrupt zip
        raise FFmpegMissing(_HELP.format(reason=f"{type(exc).__name__}: {exc}")) from exc
    return ffmpeg


def describe() -> str:
    """Where ffmpeg is coming from, for `sttop doctor`."""
    system = shutil.which("ffmpeg")
    if system:
        return f"{system} (system)"
    try:
        return f"{ffmpeg_bin()} (bundled)"
    except FFmpegMissing as exc:
        return f"missing - {exc}"
