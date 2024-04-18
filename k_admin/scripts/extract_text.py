import argparse
from pathlib import Path

from k_admin.convert_util import maybe_convert_to_txt


def main():

    args = get_cli_args()

    run(in_file=Path(args.in_file),
        encoding=args.encoding,
        clean_words=args.clean_words)


def get_cli_args() -> argparse.Namespace:
    arg_parser = argparse.ArgumentParser(description=__doc__.split('\n')[0],
                                         usage=__doc__)
    arg_parser.add_argument('in_file', help='input text or pdf file')
    arg_parser.add_argument('-e', '--encoding', help="input text file's encoding",
                            default="utf8")

    arg_parser.add_argument('-f', '--force', help="Force recreating text file, even if it already"
                                                  "exists",
                            action='store_true',
                            default=False)

    arg_parser.add_argument('-c', '--clean-words',
                            help='clean words in extracted text',
                            action='store_true')

    return arg_parser.parse_args()


def run(*, in_file: Path, encoding: str, clean_words: bool) -> None:
    # Get device
    in_txt_file = maybe_convert_to_txt(in_file, clean_words)
    if in_txt_file is None:
        return None

if __name__ == "__main__":
    main()
