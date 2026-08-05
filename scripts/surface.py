"""Compatibility command for the packaged Wisent Tools surface inspector."""

from wisent.surface import main


if __name__ == "__main__":
    raise SystemExit(main(__import__("sys").argv[1:]))
