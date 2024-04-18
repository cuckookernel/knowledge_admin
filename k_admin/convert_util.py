from pathlib import Path
from typing import Optional

from k_admin.text_util import maybe_make_checker, reconcat_lines, clean_section


def maybe_convert_to_txt(in_file: Path, clean_words: bool, force: bool = False) -> Optional[Path]:
    if in_file.suffix == '.pdf':
        in_txt_file = in_file.with_suffix('.txt')

        if not in_txt_file.exists() or force:
            print(f'Extracting text from pdf and writing to .txt file: {in_txt_file.name}')
            txt = extract_text_from_pdf(in_file, clean_words)
            in_txt_file.write_text(txt, encoding="utf-8")
        else:
            print(f"Input file is pdf but txt version ({in_txt_file}) already exists, "
                  f"not overwriting, in case txt version has been editted manually")
        return in_txt_file
    elif in_file.suffix == '.txt':
        return in_file
    else:
        print(f"ERROR: Extension of input file is {in_file.suffix}, can't only handle "
              f"PDF and TXT")
        return None


def extract_text_from_pdf(pdf_path: Path, clean_words: bool) -> str:
    # %%
    import PyPDF2
    # %%
    try:
        extracted_texts = []
        with open(pdf_path, "rb") as pdf_file:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            for page in pdf_reader.pages:
                extracted_texts.append(page.extract_text())
    except FileNotFoundError:
        return "Error: File not found."
    # %%
    extracted_text = "\n".join(extracted_texts)

    word_checker = maybe_make_checker(clean_words, extracted_text)
    # %%
    # sections = re.split("\n\n", whole_text, flags=re.MULTILINE)
    sections = reconcat_lines(extracted_text.split('\n'))

    print(f"{len(sections)} sections found:")

    clean_sections = []
    for i, section_raw in enumerate(sections):
        section = clean_section(section_raw, word_checker=word_checker)
        print(f"    {i:4d}. (len: {len(section):4d}): {section[:32]} ... {section[-32:]}")
        clean_sections.append(section)
    # %%

    return "\n".join(clean_sections)
    # %%
