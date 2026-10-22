"""Ara's frog: one silhouette, two interfaces.

The web mascot is an original SVG in ara/web/static/frog.svg. These ASCII frames
use the same proportions: wide green body, domed eyes, white throat, short legs.
The animation is real terminal output, not a single printed picture.
"""
from __future__ import annotations

import os
import shutil
import sys
import time
from typing import TextIO

GREEN = "\x1b[38;5;35m"
WHITE = "\x1b[97m"
DIM = "\x1b[2m"
RESET = "\x1b[0m"

# Seven-line frames. The frog crouches, launches, tucks its legs in the air,
# and lands with a squash before settling.
FRAMES: tuple[tuple[str, ...], ...] = (
    (
        r'      .-"""-.      ',
        r"    .'  o   o  '.    ",
        r'   /     ___     \   ',
        r'  |    /     \    |  ',
        r'  |    \_____/    |  ',
        r'   \  /|     |\  /   ',
        r'    \/ |_____| \/    ',
    ),
    (
        r'      .-"""-.      ',
        r"    .'  o   o  '.    ",
        r'   /     ___     \   ',
        r'  |    /     \    |  ',
        r'  |    \_____/    |  ',
        r'  /|\  /     \  /|\  ',
        r' /_|_\ |_____| /_|_\ ',
    ),
    (
        r'      .-"""-.      ',
        r"    .'  o   o  '.    ",
        r'   /     ___     \   ',
        r'  |    /     \    |  ',
        r' /|    \_____/    |\ ',
        r'| |   /       \   | |',
        r'\_/  /_________\_/  ',
    ),
    (
        r'    __           __    ',
        r'   /  \  .-"""-.  /  \   ',
        r'  |    || o   o ||    |  ',
        r'   \  / |  ___  | \  /   ',
        r'    \/  | /   \ |  \/    ',
        r'        \ \___/ /        ',
        r'         \_____/         ',
    ),
    (
        r'      .-"""-.      ',
        r"    .'  o   o  '.    ",
        r'   /     ___     \   ',
        r'  |    /     \    |  ',
        r'  |    \_____/    |  ',
        r'  /|   /     \   |\  ',
        r' /_|__/_______\__|_\ ',
    ),
    (
        r'     .-""""-.     ',
        r"   .'   o  o   '.   ",
        r'  /      __      \  ',
        r'  |      /  \      | ',
        r'  |      \__/      | ',
        r'   \    /    \    /  ',
        r'    \__/      \__/   ',
    ),
)

SHADOWS = ("   ~~~~~~~~~~~   ", "    ~~~~~~~~~    ", "      ~~~~~      ", "       ~~~       ", "    ~~~~~~~~~    ", "   ~~~~~~~~~~~   ")
TAGLINE = "Ara · the frog that stays on your machine"


def _color_enabled(stream: TextIO, no_color: bool | None = None) -> bool:
    if no_color is None:
        no_color = bool(os.environ.get("NO_COLOR"))
    return not no_color and bool(getattr(stream, "isatty", lambda: False)())


def static_frog(color: bool = False) -> str:
    lines = [*FRAMES[0], SHADOWS[0]]
    if color:
        return "\n".join(GREEN + line + RESET if "o" not in line else GREEN + line.replace("o", WHITE + "o" + GREEN) + RESET for line in lines)
    return "\n".join(lines)


def animation_frames() -> list[str]:
    return ["\n".join([*frame, shadow]) for frame, shadow in zip(FRAMES, SHADOWS)]


def animation_supported(stream: TextIO | None = None) -> bool:
    stream = stream or sys.stdout
    if not bool(getattr(stream, "isatty", lambda: False)()):
        return False
    if os.environ.get("ARA_NO_ANIMATION") or os.environ.get("NO_COLOR"):
        return False
    if os.environ.get("TERM", "") in {"", "dumb"}:
        return False
    return shutil.get_terminal_size((80, 24)).columns >= 30


def hop(stream: TextIO | None = None, loops: int = 2, delay: float = 0.13, color: bool | None = None) -> bool:
    """Animate a hopping frog. Returns True if it animated, False for a static print.

    Non-interactive output, dumb terminals, and ARA_NO_ANIMATION receive one static
    frog, so logs and pipes are never corrupted.
    """
    stream = stream or sys.stdout
    if not animation_supported(stream) or loops <= 0:
        stream.write(static_frog(color=_color_enabled(stream, color)) + "\n")
        stream.flush()
        return False
    use_color = _color_enabled(stream, color)
    frames = animation_frames()
    height = len(frames[0].splitlines())
    try:
        stream.write("\x1b[?25l")
        for index in range(loops * len(frames)):
            frame = frames[index % len(frames)]
            if use_color:
                painted = "\n".join(
                    GREEN + line + RESET if "~" not in line else DIM + line + RESET
                    for line in frame.splitlines()
                )
            else:
                painted = frame
            stream.write(painted + "\n")
            stream.flush()
            time.sleep(delay)
            if index != loops * len(frames) - 1:
                stream.write(f"\x1b[{height}F")
        stream.write(static_frog(color=use_color) + "\n")
        stream.flush()
        return True
    except (KeyboardInterrupt, BrokenPipeError):
        return True
    finally:
        try:
            stream.write("\x1b[?25h")
            stream.flush()
        except (OSError, ValueError):
            pass
