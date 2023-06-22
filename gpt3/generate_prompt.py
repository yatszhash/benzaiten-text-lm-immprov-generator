import openai
import music21
import numpy as np
from collections.abc import Iterable
import numpy as np
from tqdm import tqdm

from preprocess.util import extract_single_note_from_multiple_notes


def from_abc_lines_to_source_text(lines):
    """
    headerや歌詞情報は削除する。
    """
    # prompt_string = "" # codeと音の長さを示すもの
    completion_string = ""
    is_header = True
    for line in lines:
        line = line.replace("$", "")
        line = line.replace("dc=", "")
        line = line.strip()

        if line.startswith("V:"):
            is_header = False

        if is_header:
            # headerの情報を入れる
            # if line.startswith("X:"):
            #       prompt_string += line+"\n"
            if line.startswith("M:") or line.startswith("L:"):
                completion_string += line + "\n"
            elif line.startswith("L:"):
                completion_string += line + "\n"
            continue

        if line.startswith("w:"):
            continue

        if not line.startswith("V:"):
            # remove end of line comments
            index = line.rfind('%')
            if index > 0:
                line = line[:index].strip()

            # remove inline comments
            parts = line.split('"')
            newline = ""
            for i, p in enumerate(parts):
                if i % 2 == 0:
                    newline += p
                elif not p.startswith("^"):
                    newline += '"' + p + '"'
            line = ' '.join(newline.split())

            completion_string += line + "\n"
    return completion_string


def from_abc_file_to_source_line(abc_notation_filename):
    with open(abc_notation_filename) as abc_file:
        lines = abc_file.read().split('\n')

    prompt_string = process_abc_lines_to_prompt(lines)
    prompt_string = prompt_string.replace(":", ": ")
    prompt_string = prompt_string.replace('"', "`")
    prompt_string = prompt_string.replace("\n", " $ ")
    # characterでtokenizeする用にスペースを使用しない文字で置き換える。
    prompt_string = prompt_string.replace(" ", "@")
    prompt_string = ' '.join(list(prompt_string))
    # 一つの小節をsentenceとみなす
    prompt_string = prompt_string.replace("|", "</s>")

    return prompt_string


def from_abc_file_to_1line(abc_notation_filename, is_melody=False):
    with open(abc_notation_filename) as abc_file:
        lines = abc_file.read().split('\n')

    prompt_string = process_abc_lines_to_prompt(lines,
                                                keep_key=True,
                                                keep_beat=True,
                                                single_note=is_melody)
    # prompt_string = prompt_string.replace(":", ": ")
    # prompt_string = prompt_string.replace('"', "`")

    # prompt_string = prompt_string.replace("\n", " ")

    return prompt_string


def process_abc_lines_to_completion(lines):
    """
    headerや歌詞情報は削除する。
    """
    # prompt_string = "" # codeと音の長さを示すもの
    completion_string = ""  # melody
    is_header = True
    for line in lines:
        line = line.replace("$", "")
        line = line.replace("dc=", "")
        line = line.strip()

        if line.startswith("V:"):
            is_header = False

        if is_header:
            # headerの情報を入れる
            # if line.startswith("X:"):
            #       prompt_string += line+"\n"
            continue

        if line.startswith("w:"):
            continue

        if not line.startswith("V:"):
            # remove end of line comments
            index = line.rfind('%')
            if index > 0:
                line = line[:index].strip()

            # remove inline comments
            parts = line.split('"')
            newline = ""
            for i, p in enumerate(parts):
                if i % 2 == 0:
                    newline += p
                elif not p.startswith("^"):
                    newline += '"' + p + '"'
            line = ' '.join(newline.split())

            completion_string += line + "\n"
    return completion_string


def process_abc_lines_to_prompt(lines,
                                keep_key=False,
                                keep_beat=False,
                                single_note=False):
    """
    headerや歌詞情報は削除する。
    """
    prompt_string = ""  # codeと音の長さを示すもの
    is_header = True
    for line in lines:
        line = line.replace("$", "")
        line = line.replace("dc=", "")
        line = line.strip()

        if line.startswith("V:"):
            is_header = False

        if is_header:
            # headerの情報を入れる
            # if line.startswith("X:"):
            #       prompt_string += line+"\n"
            if keep_key and line.startswith("K"):
                prompt_string += line + "\n"
                continue

            if keep_beat and (line.startswith("L") or line.startswith('M')):
                prompt_string += line + "\n"
                continue
            continue

        if line.startswith("w:"):
            continue

        if not line.startswith("w:") and not line.startswith("V:"):
            # remove end of line comments
            index = line.rfind('%')
            if index > 0:
                line = line[:index].strip()

            # remove inline comments
            # parts = line.split('"')
            # newline = ""
            # for i, p in enumerate(parts):
            #     if i%2 == 0:
            #         newline += p
            #     elif not p.startswith("^"):
            #         newline += '"' + p + '"'
            # line = ' '.join(newline.split())

            # 改行は音楽的要素と関係ないため入れない
            prompt_string += ' ' + line
    if single_note:
        prompt_string = extract_single_note_from_multiple_notes(prompt_string)
    return prompt_string


def to_prompt_completion(chord_abc_noation_filename,
                         melody_abc_notation_filename):
    showed_title = False

    with open(chord_abc_noation_filename) as song_file:
        chord_lines = song_file.read().split('\n')

    with open(melody_abc_notation_filename) as song_file:
        melody_lines = song_file.read().split('\n')

    prompt_string = process_abc_lines_to_prompt(chord_lines)
    prompt_string = prompt_string.replace(":", ": ")
    prompt_string = prompt_string.replace('"', "`")
    prompt_string = prompt_string.replace("\n", " $ ")

    completion_string = process_abc_lines_to_completion(melody_lines)
    completion_string = completion_string.replace('"', "`")
    completion_string = completion_string.strip().replace("\n", " $ ")

    prompt = '{"prompt": "' + prompt_string + '<song>", '
    prompt += '"completion": " ' + completion_string + ' $ <end>"}\n'

    #     if prompt not in prompts:
    #       original_songs.append(s)
    #       prompt_file.write(prompt)
    #       prompts.append(prompt)
    #       num_prompts += 1
    return prompt


def generate_to_prompt_from_abc_files(segmetn_abc_notation_root):
    prompts = set()
    for melody_path in tqdm(
            sorted(segmetn_abc_notation_root.glob('**/melody.abc'))):
        print(melody_path)
        chord_path = melody_path.with_stem('chord')
        if not chord_path.exists():
            print('chord file not found')
            continue
        prompts.add(to_prompt_completion(chord_path, melody_path))
    return prompts


def generate_inference_prompt_from_file(chord_abc_noation_filename):
    showed_title = False

    with open(chord_abc_noation_filename) as song_file:
        chord_lines = song_file.read().split('\n')

    # with open(melody_abc_notation_filename) as song_file:
    #     melody_lines = song_file.read().split('\n')

    prompt_string = process_abc_lines_to_prompt(chord_lines)
    prompt_string = prompt_string.replace(":", ": ")
    prompt_string = prompt_string.replace('"', "`")
    prompt_string = prompt_string.replace("\n", " $ ")

    prompt = '{"prompt": "' + prompt_string + '<song>}'

    #     if prompt not in prompts:
    #       original_songs.append(s)
    #       prompt_file.write(prompt)
    #       prompts.append(prompt)
    #       num_prompts += 1
    return prompt
