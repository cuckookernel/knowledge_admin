#!/bin/env python
"""Turn a long text into a sound file (wav or mp3), using Coqui TTS: text-to-speech

Example usages:

# Generate a wav file from a plain text document (assumed to be utf-8 encoded by default)
# Path to wav file will my_dir/my_document.wav
./doc_2_audio.py my_dir/my_document.txt

# Will produce output file: my_dir/my_document.wav

# Generate a wav file from a plain text document encoded in utf-16
# (all encodings supported by Python open() are supported here, e.g. cp1252, iso-8859-1, etc.)
./doc_2_audio.py -e utf-16 my_dir/my_document.txt

# Generate a wav file from a PDF document (requires PyPDF2 Python library to be installed)
./doc_2_audio.py  my_dir/my_document.pdf

# Generate a wav file from a PDF document and wrongly split/concatenated words
# (requires PyPDF2 and enchant Python libraries to be installed)
./doc_2_audio.py -c my_dir/my_document.pdf

# Generate an mp3 file from a PDF
# (requires PyPDF2 and pydub Python libraries)
./doc_2_audio.py -f mp3  my_dir/my_document.pdf

# Will produce output file: my_dir/my_document.mp3

For other options:
./doc_2_audio.py -h

Requirements:
pip install TTS  # install coqui-tts library for text to speech conversion
pip install PyPDF2  # only needed if input doc is going to be in PDF format
pip install pyenchant # only needed if passing --clean-words
pip install pydub # only needed if passing

It could be a good idea to create venv first (before pip install):
python3 -m venv venv-d2a; venv-d2a/bin/activate.sh; pip install wheel

Todo:
----
- Automatically extract plain text from a pdf document.
- Generate mp3 instead of wav

"""

import numpy as np
import argparse
from pathlib import Path

import torch
from TTS.api import TTS
from TTS.utils.audio.numpy_transforms import save_wav

from kb_mgmt.text_util import clean_word, reconcat_lines, read_text_file, reconcat_lines_v2
from kb_mgmt.convert_util import maybe_convert_to_txt, extract_text_from_pdf

TTS_MODELS_BY_KEY = {
    'vits': 'tts_models/en/vctk/vits'
}
# %%


def main():

    args = get_cli_args()

    run(in_file=Path(args.in_file),
        encoding=args.encoding,
        clean_words=args.clean_words,
        concat_policy=args.concat_policy,
        tts_model_key=args.tts_model_key,
        speaker=args.speaker,
        out_fmt=args.out_fmt)


def get_cli_args() -> argparse.Namespace:
    arg_parser = argparse.ArgumentParser(description=__doc__.split('\n')[0],
                                         usage=__doc__)
    arg_parser.add_argument('in_file', help='input text or pdf file')
    arg_parser.add_argument('-e', '--encoding', help="input text file's encoding",
                            default="utf8")
    arg_parser.add_argument('-k', '--tts_model_key',
                            help=f'tts model key, possible values: '
                                 f'{list(TTS_MODELS_BY_KEY.keys())}',
                            default="vits")
    arg_parser.add_argument('-s', '--speaker',
                            help='speaker used in the case of a multi-speaker model',
                            default="p225")

    arg_parser.add_argument('-c', '--clean-words',
                            help='clean words in extracted text',
                            action='store_true')

    arg_parser.add_argument('-l', '--concat-policy',
                            help='policy for concatting lines',
                            default="v2")

    arg_parser.add_argument('-f', '--out-fmt',
                            help='the format of the output, either `mp3` or `wav`',
                            default='wav')

    return arg_parser.parse_args()


def run(*, in_file: Path, encoding: str,
        clean_words: bool, concat_policy: str,
        tts_model_key: str,
        speaker: str, out_fmt: str) -> None:
    # Get device
    tts_model_name = TTS_MODELS_BY_KEY[tts_model_key]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tts_model = TTS(tts_model_name).to(device)

    in_txt_file = maybe_convert_to_txt(in_file, clean_words)
    if in_txt_file is None:
        return None

    sections = read_text_file(in_txt_file, encoding=encoding)
    if concat_policy == "v2":
        sections = reconcat_lines_v2(sections)
    # text = "\n".join(sections)

    out_wav_file = in_txt_file.with_suffix(".wav")
    tts_by_sections(tts_model, sections=sections, speaker=speaker, out_wav_path=out_wav_file)

    if out_fmt == 'mp3':
        produce_mp3_output(out_wav_file)
        out_wav_file.unlink()
    else:
        print(f"Done generating wav file: {out_wav_file}")


# %%


def tts_by_sections(tts_model: TTS, *, speaker: str, sections: list[str], out_wav_path: Path):

    n_sections = len(sections)
    total_len = sum(len(section) for section in sections)
    processed_len = 0
    wavs: list[np.ndarray] = []

    sections_ = [section for section in sections if len(section.strip()) > 0]
    if len(sections_) < len(sections):
        print(f'Discarded {len(sections) - len(sections_)} empty sections.')
    del sections

    for i, section in enumerate(sections_):
        wav_list = tts_model.tts(text=section, speaker=speaker)
        processed_len += len(section)

        wav = np.array(wav_list)
        print(f"Done with section {i:4d} / {n_sections}, progress: "
              f"{processed_len / total_len * 100:3.1f} % \n")
        wavs.append(wav)
        # TODO: add pause between consecutive wavs ?

    final_wav = np.hstack(wavs)
    final_len_secs: float = len(final_wav) / tts_model.synthesizer.output_sample_rate
    print(f'Saving final wav with {len(final_wav)} samples '
          f'({final_len_secs:.2f} s) to: {out_wav_path}')
    save_wav(wav=final_wav, path=str(out_wav_path),
             sample_rate=tts_model.synthesizer.output_sample_rate)
# %%


def produce_mp3_output(out_wav_file: Path):
    from pydub import AudioSegment

    audio = AudioSegment.from_wav(out_wav_file)
    out_mp3_file = out_wav_file.with_suffix('.mp3')
    audio.export(out_mp3_file, format='mp3')
    print(f"Done generating mp3 file: {out_mp3_file}")


def interactive_testing():
    # %%
    runfile("tts/doc_2_audio.py")

    # %%
    pdf_path = Path("/home/teo/gdrive_rclone/Academico/MAIA/Etica de la IA/Week 1/"
                    "1.7-Why-Teaching-ethics-to-AI-practitioners-is-important.pdf")
    # %%
    pdf_text = extract_text_from_pdf(pdf_path, clean_words=True)
    print(pdf_text[:10])
    # Print the extracted text
    # %%
    lines = read_text_file(pdf_path.with_suffix('.txt'))
    # text = pdf_path.with_suffix('.txt').read_text(encoding='utf-8')

    for i, line in enumerate(lines):
        print(f"{i:5d} | {line}")

    # %%
    sections = reconcat_lines(lines)

    for i, line in enumerate(sections):
        print(f"{i:5d} | {line}")

    # %%
    txt_fpath = Path("/home/teo/Downloads/MAIA/Etica-AI/"
                     r"Etzioni - Etzioni - Incorporating Ethics into AI.txt")
    sections = read_text_file(txt_fpath)
    print(f'raw sections: {len(sections)}')
    # %%
    sections2 = reconcat_lines_v2(sections)
    print(f'after concat sections: {len(sections2)}')

    # %%
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tts_model = TTS('tts_models/en/vctk/vits').to(device)
    # wav = tts.tts(sections[35], speaker='p225')
    # %%
    tts_by_sections(tts_model, speaker='p225', sections=sections[0: 10],
                    out_wav_path=Path('./test.wav'))
    # %%
    import enchant

    word_checker = enchant.Dict('en')

    result = clean_word('fr-om', word_checker=word_checker)
    print(result)
    # %%


if __name__ == '__main__':
    main()
