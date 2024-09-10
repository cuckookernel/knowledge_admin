#!/bin/env python
"""Turn a long text into a sound file (wav or mp3), using Coqui TTS: text-to-speech

Example usages:

# Generate a wav file from a plain text document (assumed to be utf-8 encoded by default)
# Path to wav file will my_dir/my_document.wav
doc_2_audio my_dir/my_document.txt

# Will produce output file: my_dir/my_document.wav

# Generate a wav file from a plain text document encoded in utf-16
# (all encodings supported by Python open() are supported here, e.g. cp1252, iso-8859-1, etc.)
doc_2_audio -e utf-16 my_dir/my_document.txt

# Generate a wav file from a PDF document (requires PyPDF2 Python library to be installed)
doc_2_audio.py  my_dir/my_document.pdf

# Generate a wav file from a PDF document and wrongly split/concatenated words
# (requires PyPDF2 and enchant Python libraries to be installed)
doc_2_audio -c my_dir/my_document.pdf

# Generate an mp3 file from a PDF
# (requires PyPDF2 and pydub Python libraries)
doc_2_audio -f mp3  my_dir/my_document.pdf

# Will produce output file: my_dir/my_document.mp3

For other options:
doc_2_audio -h

Extra Requirements:
pip install PyPDF2  # if input doc is in PDF format
pip install pyenchant # only needed if using --clean-words option
pip install pydub # only needed if passing -f mp3

It could be a good idea to create venv first (before pip install):
python3 -m venv venv-k-admin; venv-k-admin/bin/activate.sh; pip install wheel

"""
from typing import Optional

import numpy as np
import argparse
from pathlib import Path

import torch
from TTS.api import TTS
from TTS.utils.audio.numpy_transforms import save_wav

from k_admin.text_util import \
    clean_word, reconcat_lines, read_text_file, reconcat_lines_v2, reconcat_lines_v3
from k_admin.convert_util import maybe_convert_to_txt, extract_text_from_pdf

ModelKey: type = str  # a short model key for dictionary below
ModelName: type = str  # the fully "qualified" model name, which looks like a path: tts_models/...
LangCode: type = str  # a two letter language code
SpeakerId: type = str

# mapping of model key to modelname
DEFAULT_MODEL_NAME_BY_LANG: dict[LangCode, ModelName] = {
    'en': 'tts_models/en/vctk/vits',
    'es': 'tts_models/es/css10/vits',
    # 'es': 'tts_models/multilingual/multi-dataset/xtts_v2' ## SLOW...
}

TTS_MODEL_NAME_BY_KEY: dict[ModelKey, ModelName] = {
    'vits': 'tts_models/en/vctk/vits',
    # Good quality but slow....
    'xtts_v2': 'tts_models/multilingual/multi-dataset/xtts_v2',
}

# mapping of (Model, Language) -> Speaker Name
DEFAULT_SPEAKER_FOR_MODEL_LANG: dict[tuple[ModelKey, LangCode], Optional[SpeakerId]] = {
    ('tts_models/en/vctk/vits', 'en'): 'p225',
    # p234 male - british
    # p229 M - slow -clear
    # p250 F - fast
    # p251 M - fast
    # p376 male- robotic - noisy
    ('tts_models/multilingual/multi-dataset/xtts_v2', 'es'): 'Gilberto Mathias',
    ('tts_models/es/css10/vits', 'es'): None
}
# %%
# 'p225': 1, 'p226': 2, 'p227': 3, 'p228': 4, 'p229': 5, 'p230': 6, 'p231': 7, 'p232': 8, 'p233': 9,
# 'p234': 10, 'p236': 11, 'p237': 12, 'p238': 13, 'p239': 14, 'p240': 15, 'p241': 16, 'p243': 17,
# 'p244': 18, 'p245': 19, 'p246': 20, 'p247': 21, 'p248': 22, 'p249': 23, 'p250': 24,
# 'p251': 25, 'p252': 26, 'p253': 27, 'p254': 28, 'p255': 29, 'p256': 30, 'p257': 31,
# 'p258': 32, 'p259': 33, 'p260': 34, 'p261': 35, 'p262': 36, 'p263': 37, 'p264': 38,
# 'p265': 39, 'p266': 40, 'p267': 41, 'p268': 42, 'p269': 43, 'p270': 44, 'p271': 45, 'p272': 46,
# 'p273': 47, 'p274': 48, 'p275': 49, 'p276': 50, 'p277': 51, 'p278': 52, 'p279': 53, 'p280': 54,
# 'p281': 55, 'p282': 56, 'p283': 57, 'p284': 58, 'p285': 59, 'p286': 60, 'p287': 61, 'p288': 62,
# 'p292': 63, 'p293': 64, 'p294': 65, 'p295': 66, 'p297': 67, 'p298': 68, 'p299': 69, 'p300': 70,
# 'p301': 71, 'p302': 72, 'p303': 73, 'p304': 74, 'p305': 75, 'p306': 76, 'p307': 77, 'p308': 78,
# 'p310': 79, 'p311': 80, 'p312': 81, 'p313': 82, 'p314': 83, 'p316': 84, 'p317': 85, 'p318': 86,
# 'p323': 87, 'p326': 88, 'p329': 89, 'p330': 90, 'p333': 91, 'p334': 92, 'p335': 93, 'p336': 94,
# 'p339': 95, 'p340': 96, 'p341': 97, 'p343': 98, 'p345': 99, 'p347': 100, 'p351': 101, 'p360': 102,
# 'p361': 103, 'p362': 104, 'p363': 105, 'p364': 106, 'p374': 107, 'p376': 108}


def main():

    args = get_cli_args()

    run(in_file=Path(args.in_file),
        encoding=args.encoding,
        clean_words=args.clean_words,
        concat_policy=args.concat_policy,
        tts_model_key=args.tts_model_key,
        language=args.language,
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
                                 f'{list(TTS_MODEL_NAME_BY_KEY.keys())}',
                            default=None)
    arg_parser.add_argument('-t', '--language',
                            help=f'two-letter code for language, e.g "en" or "es"',
                            default="en"),

    arg_parser.add_argument('-s', '--speaker',
                            help='speaker used in the case of a multi-speaker model',
                            default=None)

    arg_parser.add_argument('-c', '--clean-words',
                            help='clean words in extracted text',
                            action='store_true')

    arg_parser.add_argument('-l', '--concat-policy',
                            help='policy for concatting lines, possible values v2, v3',
                            default="v3")

    arg_parser.add_argument('-f', '--out-fmt',
                            help='the format of the output, either `mp3` or `wav`',
                            default='wav')

    return arg_parser.parse_args()


def run(*, in_file: Path, encoding: str,
        clean_words: bool, concat_policy: str,
        language: Optional[str],
        tts_model_key: Optional[str],
        speaker: str, out_fmt: str) -> None:
    # Get device
    tts_model_path = determine_model_path(tts_model_key, language)

    speaker = determine_speaker(tts_model_path, language, speaker)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"INFO: using device={device}, building TTS model now")

    tts_model = TTS(tts_model_path).to(device)

    language = validate_language(tts_model, language)

    in_txt_file = maybe_convert_to_txt(in_file, clean_words)
    if in_txt_file is None:
        return None

    sections = read_text_file(in_txt_file, encoding=encoding)
    if concat_policy == "v2":
        sections = reconcat_lines_v2(sections)
    elif concat_policy == "v3":
        sections = reconcat_lines_v3(sections)
    # text = "\n".join(sections)

    out_wav_file = in_txt_file.with_suffix(".wav")
    tts_by_sections(tts_model, language=language, speaker=speaker,
                    sections=sections,  out_wav_path=out_wav_file)

    if out_fmt == 'mp3':
        produce_mp3_output(out_wav_file)
        out_wav_file.unlink()
    else:
        print(f"Done generating wav file: {out_wav_file}")


def determine_model_path(tts_model_key: ModelKey | None, language: LangCode | None) -> str:
    if tts_model_key is None:
        if language is None:
            exit_error("Need to specify either tts_model_key or language")
            return "Ignored"
        else:
            tts_model_path = DEFAULT_MODEL_NAME_BY_LANG[language]
            print(f"INFO: Selected default model path='{tts_model_path}'"
                  f"for based on lang={language}")
    else:
        tts_model_path = TTS_MODEL_NAME_BY_KEY[tts_model_key]
        print(f"INFO: model path='{tts_model_path}', mapped from tts_model_key={tts_model_key}")

    return tts_model_path


def determine_speaker(tts_model_name: ModelName, language: LangCode, speaker: str) -> str:
    if speaker is None:
        key = (tts_model_name, language)
        if key not in DEFAULT_SPEAKER_FOR_MODEL_LANG:
            exit_error(f"The default speaker defined for (model_name, lang) = {key}")
        speaker = DEFAULT_SPEAKER_FOR_MODEL_LANG[key]
        print(f"INFO: Selected default speaker='{speaker}' for {key}")

    return speaker


def validate_language(tts_model, language) -> str:
    if tts_model.languages is not None:
        if language not in tts_model.languages:
            print(f"ERROR: language='{language}' not in model.languages for "
                  f"tts_model={tts_model.name}, "
                  f"run tts --list_models to find other models that might "
                  f"support this language")
    else:  # model is not multilanguage
        language = None

    return language


def tts_by_sections(tts_model: TTS, *, speaker: Optional[str], language: str,
                    sections: list[str], out_wav_path: Path):

    n_sections = len(sections)
    total_len = sum(len(section) for section in sections)
    processed_len = 0
    wavs: list[np.ndarray] = []

    sections_ = [section for section in sections if len(section.strip()) > 0]
    if len(sections_) < len(sections):
        print(f'Discarded {len(sections) - len(sections_)} empty sections.')
    del sections

    for i, section in enumerate(sections_):
        wav_list = tts_model.tts(text=section, speaker=speaker, language=language)
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


def exit_error(msg: str):
    print(f"ERROR: {msg}")
    exit(1)


def interactive_testing():
    # %%
    runfile("k_admin/scripts/doc_2_audio.py")

    # %%
    pdf_path = Path("/home/teo/gdrive_rclone/Academico/MAIA/Etica de la IA/Week-4/"
                    "1.7-Why-Teaching-ethics-to-AI-practitioners-is-important.pdf")
    # %%
    tts = TTS('tts_models/multilingual/multi-dataset/xtts_v2')
    # %%

    print("is_multi_speaker: ", tts.is_multi_speaker)
    # %%
    tts.speaker_manager.speakers
    # %%
    tts_by_sections(tts, sections=["Un algoritmo puede ser opaco en dos sentidos muy diferentes. "
                                   "Por una parte, algunos algoritmos son llamados “cajas negras” "
                                   "porque son secretos industriales protegidos por "
                                   "leyes de propiedad intelectual."],
                    # speaker="Alma María",
                    speaker='Gilberto Mathias',
                    language="es",
                    out_wav_path=Path("test.wav"))
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
