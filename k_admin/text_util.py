import re
from pathlib import Path
from typing import Protocol, Optional


class WordChecker(Protocol):
    def check(self, word: str) -> bool:
        pass


def clean_section(section_raw: str, word_checker: Optional[WordChecker]) -> str:
    # %%
    lines = section_raw.split("\n")
    new_lines = [clean_line(line, word_checker) for line in lines]
    # %%
    return " ".join(new_lines)


def clean_line(line: str, word_checker: Optional[WordChecker]) -> str:
    # %%
    clean_line = line.replace('ﬁ', 'fi').strip()
    if word_checker is None:
        return clean_line
    else:
        words = clean_line.split()
        return " ".join(clean_word(word, word_checker) for word in words)


def clean_word(word: str, word_checker: WordChecker) -> str:
    if len(word) == 0:
        return word

    if word[-1] in ',;?.)(!':
        return clean_word(word[:-1], word_checker) + word[-1]

    if '-' in word:
        no_dash = word.replace('-', '')
        if len(no_dash) > 0 and word_checker.check(no_dash):
            # print(f'removed dash: `{word}` -> {no_dash}')
            return no_dash
        else:
            return word
    else:
        if word_checker.check(word):
            return word
        else:

            # print(f'Word: `{word} not in dict.', end=' ')
            for i in range(1, len(word) - 1):
                part1 = word[:i]
                part2 = word[i:]

                if word_checker.check(part1) and word_checker.check(part2):
                    # print(f'split word: {part1} {part2}')
                    return f'{part1} {part2}'

            return word


def auto_detect_language(text: str, candidate_langs: list[str] = None) -> str:
    words = [w for w in re.split(r'\W', text) if len(w) > 0]

    def score_lang(lang: str) -> float:
        import enchant
        lang_dict = enchant.Dict(lang)
        n_words_in_dict = sum(lang_dict.check(word)
                              for word in words if lang_dict.check(word))
        score = -n_words_in_dict / len(words) * 100.0
        print(f"score for {lang}: {score:.2f}")

        return score

    if candidate_langs is None:
        candidate_langs = ['en', 'es']

    assert len(candidate_langs) > 0
    langs_by_score = sorted(candidate_langs, key=score_lang)

    return langs_by_score[0]


def reconcat_lines(lines: list[str]) -> list[str]:
    """Process lines so that a line ending in '-'
    is concatted to the next line (if starting with a lowercase letter)"""
    out_lines = []
    curr_line = ''

    for line in lines:
        if curr_line.endswith('-') and line[0].lower() == line[0]:
            curr_line = curr_line[:-1] + line
        elif len(line) > 0 and line[0].lower() == line[0]:
            curr_line += ' ' + line
        else:
            if curr_line != '':
                out_lines.append(curr_line)
            curr_line = line

    out_lines.append(curr_line)

    return out_lines


def reconcat_lines_v2(lines: list[str]) -> list[str]:
    out_paraphs = []

    curr_buffer = []
    for l_idx, line in enumerate(lines):
        is_title = re.search('^[0-9]', line) is not None and len(curr_buffer) == 0
        ends_in_punctuation = re.search(r'[.?!]\s*$', line) is not None
        print(f'{l_idx} |{line}|\nis_title:{is_title} ends_in_punct:{ends_in_punctuation}')
        if is_title or ends_in_punctuation:
            curr_buffer.append(line)
            new_paraph = ' '.join(curr_buffer)
            print(f"{len(out_paraphs)} | {new_paraph}")
            curr_buffer = []
            out_paraphs.append(new_paraph)
        else:
            curr_buffer.append(line)

    new_paraph = ' '.join(curr_buffer)
    # print(f"{len(out_paraphs)} | {new_paraph}")
    out_paraphs.append(new_paraph)

    return out_paraphs
# %%


def reconcat_lines_v3(lines: list[str]) -> list[str]:
    out_paraphs = []

    curr_buffer = []
    for l_idx, line in enumerate(lines):
        is_title = re.search('^[0-9]', line) is not None and len(curr_buffer) == 0
        ends_in_punctuation = re.search(r'[.?!]\s*$', line) is not None
        print(f'{l_idx} |{line}|\nis_title:{is_title} ends_in_punct:{ends_in_punctuation}')
        if is_title or ends_in_punctuation:
            curr_buffer.append(line)
            new_paraph = build_paraph(curr_buffer)
            print(f"{len(out_paraphs)} | {new_paraph}")
            curr_buffer = []
            out_paraphs.append(new_paraph)
        else:
            curr_buffer.append(line)

    new_paraph = build_paraph(curr_buffer)
    # print(f"{len(out_paraphs)} | {new_paraph}")
    out_paraphs.append(new_paraph)

    return out_paraphs


def build_paraph(curr_buffer: list[str]) -> str:
    """Build a single string from a list of lines
    removing '-' endings from lines and inserting spaces where appropriate"""

    pieces = []
    for i, line in enumerate(curr_buffer):
        line = line.strip()
        if line.endswith('-'):
            line = line[:-1]
            pieces.append(line)
        else:
            pieces.append(line)
            pieces.append(' ')

    return ''.join(pieces)


def maybe_make_checker(clean_words: bool, extracted_text: str) -> Optional[WordChecker]:
    import enchant
    lang = auto_detect_language(extracted_text)
    return enchant.Dict(lang) if clean_words else None


def read_text_file(in_txt_file: Path, encoding: str = "utf8") -> list[str]:
    """Return one string for each section"""
    whole_text = in_txt_file.read_text(encoding=encoding)

    sections = whole_text.split('\n')
    return sections


def count_words(text: str) -> int:
    parts = re.split(r"\b", text)

    return sum(1 if not is_space(a_str) else 0
               for a_str in parts)


def is_space(a_str: str) -> bool:
    return a_str == ' ' or re.match(r'[,.:?()!]?\s+', a_str)
