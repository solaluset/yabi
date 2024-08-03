import sys
import argparse

from .parser import to_pure_python, to_bython


parser = argparse.ArgumentParser(
    description="Python <==> Bython converter",
)
parser.add_argument(
    "--to-python",
    dest="to_python",
    action="store_true",
    help="convert Bython to Python",
)
parser.add_argument("target", type=argparse.FileType())


def convert_main(args=sys.argv[1:]):
    args = parser.parse_args(args)
    converter = to_pure_python if args.to_python else to_bython
    print(converter(args.target.read()).rstrip("\n"))
