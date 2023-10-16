from auto_eval import __app_name__

from . import cli


def main():
    cli.app(prog_name=__app_name__)


if __name__ == "__main__":
    main()
