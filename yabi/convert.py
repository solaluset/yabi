import argparse
import sys

from .parser import to_bython, to_pure_python

parser = argparse.ArgumentParser(
    description="Python <==> Bython converter",
)
parser.add_argument(
    "--to-python",
    dest="to_python",
    action="store_true",
    help="convert Bython to Python",
)
parser.add_argument("target")


def main(args=sys.argv[1:]):
    args = parser.parse_args(args)
    converter = to_pure_python if args.to_python else to_bython
    with open(args.target) as source:
        print(converter(source.read()).rstrip("\n"))


if __name__ == "__main__":
    main()
