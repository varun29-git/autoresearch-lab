"""Protected experiment runner. Run `python train.py --help`."""

from research.loop import main


if __name__ == "__main__":
    main(default_command="trial")
