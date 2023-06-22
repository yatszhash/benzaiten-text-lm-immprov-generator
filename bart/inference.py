import copy
import logging
import os
import pathlib
import random
import re
import subprocess
import sys

import fairseq
import matplotlib.pyplot as plt
import midi2audio
import mido
import music21
import numpy
import pandas as pd
import pypianoroll

from gpt3.generate_prompt import from_abc_file_to_1line, from_abc_file_to_source_line
from gpt3.inference import from_chord_csv_to_chord_score
from preprocess.util import replace_rest_to_chord_notes, to_abc_notation

OCTAVES = [2]

MELODY_VELOCITY = 127

TOTAL_MEASURES = 240  # 学習用MusicXMLを読み込む際の小節数の上限
UNIT_MEASURES = 4  # 1回の生成で扱う旋律の長さ
BEAT_RESO = 4  # 1拍を何個に分割するか（4の場合は16分音符単位）
N_BEATS = 4  # 1小節の拍数（今回は4/4なので常に4）
NOTENUM_FROM = 36  # 扱う音域の下限（この値を含む）
NOTENUM_THRU = 84  # 扱う音域の上限（この値を含まない）
INTRO_BLANK_MEASURES = 4  # ブランクおよび伴奏の小節数の合計
MELODY_MEASURE_LENGTH = 8  # 生成するメロディの長さ（小節数）
MIDI_RESOLUTION = 480

KEY_ROOT = "C"  # 生成するメロディの調のルート（"C" or "A"）
KEY_MODE = "major"  # 生成するメロディの調のモード（"major" or "minor"）
FLUIDSYNTH_PATH = '/opt/homebrew/bin/fluidsynth'

# defaultが1024で合わないので、書き換えておく
music21.defaults.ticksPerQuarter = MIDI_RESOLUTION
logger = logging.getLogger(__name__)


def from_abc_to_midi(abc_file, output_midi_path, tempo=120):
    output_bytes = subprocess.check_output([
        os.getenv('ABC2MIDI'),
        str(abc_file), "-v", "-o",
        str(output_midi_path), "-Q",
        str(tempo)
    ],
                                           timeout=20)
    print(output_bytes)
    # output = output_bytes.decode("utf-8").strip()
    # with open(output_file, "w") as abc_file:
    #     abc_file.write(output)
    # except:
    #     print("Unexpected error:", sys.exc_info()[0])


def extract_bpm(midifile: mido.MidiFile):
    meta_track = midifile.tracks[0]
    for message in meta_track:
        if message.type == 'set_tempo':
            return mido.tempo2bpm(message.tempo)
    raise ValueError('tempo not defined')


def extract_key_signature(midifile: mido.MidiFile):
    meta_track = midifile.tracks[0]
    for message in meta_track:
        if message.type == 'key_signature':
            return message.key
    raise ValueError('tempo not defined')


def generate_octave(input_midi_path, output_midi_path, num_octave=1):
    midi_file = mido.MidiFile(input_midi_path)
    for message in midi_file.tracks[0]:
        if message.type in ('note_on', 'note_off'):
            message.note += 12 * num_octave
    midi_file.save(output_midi_path)


SCALES = {
    'blues': {
        'C': [0, 3, 5, 6, 7, 10],  # C	Eb	F	Gb	G	Bb
        'Amin': [0, 2, 3, 4, 7, 9]  # C – D – Eb – E – G - A
    },
    # Aminorは経験上ブルースにしておく
    'diatonic': {
        'C': [0, 2, 4, 5, 7, 9, 11],
        # 'Amin': [0, 2, 4, 5, 7, 9, 11]
        'Amin': [0, 2, 3, 4, 7, 9]
    },
    'melodic_minor': {
        'C': [0, 2, 3, 5, 7, 9, 11],
        # 'Amin': [0, 2, 4, 6, 8, 9, 11]
        'Amin': [0, 2, 3, 4, 7, 9]
    },
    'dorian': {
        'C': [0, 2, 3, 5, 7, 9, 10],
        'Amin': [0, 2, 4, 6, 7, 9, 11]
    }
}

random.seed(0)

# def find_nearest_scale_pitch(pitch, scale):
#     if pitch <= scale[0]:
#         return scale[0]
#     if pitch >= scale[-1]:
#         return scale[-1]
#
#     return


def to_scale_melody(note, scale_name: str, key):
    pitch = note % 12
    transpose = note // 12
    scale_notes = SCALES[scale_name][key]
    if pitch in scale_notes:
        return note
    else:  # それ以外の場合は，scale上のpitchに修正 randomnessは生成時に確保されていると思われるので決定的に置き換える。
        find_index = numpy.searchsorted(scale_notes,
                                        note,
                                        side='left',
                                        sorter=None)
        if find_index < len(scale_notes):
            nearest_pitch = scale_notes[find_index]
        else:
            nearest_pitch = scale_notes[find_index - 1]
        # if random.random() < 0.5:
        #     nearest_pitch = note.pitch - 1
        # else:
        #     nearest_pitch = note.pitch + 1
        nearest_pitch = nearest_pitch + transpose * 12
        return nearest_pitch


#
# # 最新の生成ファイルを指定（どれを選ぶかは自由。例：人間の耳で確認して良いものを指定する）
# melody_filepaths = glob.glob(basedir + '/output/*.mid')  # 全ての出力メロディのリストを取得
# melody_filepaths.sort(reverse=True)
# melody_filepath = melody_filepaths[0]
# print(melody_filepath)
# midi_data = pretty_midi.PrettyMIDI(melody_filepath)
#
# for instrument in midi_data.instruments:
#     # TODO: ここでメロディパートなのか伴奏パートなのかを区別する必要がある
#     for note in instrument.notes:
#         start, end = int(note.start * MELODY_LENGTH), int(
#             note.end * MELODY_LENGTH)  # ループ内で見ているnoteの開始点と終了点を算出
#         modified_pitch = postprocess_to_diatonic_melody(note)
#         pianoroll[start:end, :] = np.zeros(NOTENUM_THRU - NOTENUM_FROM + 1, )
#         while modified_pitch - NOTENUM_FROM > pianoroll.shape[
#                 1]:  # 高い音を出しすぎた場合はオク下げ
#             modified_pitch -= 12
#
#         pianoroll[start:end, modified_pitch - NOTENUM_FROM] = 1
#
# show_and_play_midi(pianoroll, 12, basedir + "/" + backing_file,
#                    basedir + output_file)


def replace_note_with_scale_note(midi_path,
                                 scale_name,
                                 key,
                                 replace_only_long=False):
    midi = mido.MidiFile(midi_path)

    replaced_note_on = None
    replacer = {}
    for i, message in enumerate(midi.tracks[0]):
        if message.type == 'note_on':
            note_duration = midi.tracks[0][i + 1].time - message.time + 1
            if not replace_only_long or note_duration >= midi.ticks_per_beat:
                replaced_note_on = to_scale_melody(note=message.note,
                                                   scale_name=scale_name,
                                                   key=key)
                if replaced_note_on != message.note:
                    new_message = copy.deepcopy(message)
                    new_message.note = replaced_note_on
                    replacer[i] = new_message
        if message.type == 'note_off' and replaced_note_on:
            new_message = copy.deepcopy(message)
            new_message.note = replaced_note_on
            replacer[i] = new_message
            replaced_note_on = None

    for i, message in replacer.items():
        midi.tracks[0][i] = message

    midi.save(midi_path)


def append_end_note(midi_file: pathlib.Path, key):
    key_base_note = 0 if key == 'C' else 9  # Amの場合
    chord_I_notes = [4] if key == 'C' else []  # Amの場合
    midi = mido.MidiFile(midi_file)

    tempo = None
    for message in midi.tracks[0]:
        if message.type == 'set_tempo':
            tempo = message.tempo
            break

    # 誤差がでるので少しだけ長めにしておく。
    melody_duration = mido.tick2second(
        midi.ticks_per_beat * (MELODY_MEASURE_LENGTH + INTRO_BLANK_MEASURES) *
        N_BEATS, midi.ticks_per_beat, tempo) + mido.second2tick(
            0.03, midi.ticks_per_beat, tempo)
    # たまに長すぎるmidiが生成されるのでその場合はlast noteを削除する。
    while midi.length >= melody_duration:
        del midi.tracks[0][-3]
        del midi.tracks[0][-3]

    melody_last_note = midi.tracks[0][-3].note
    last_note_init_tick = (INTRO_BLANK_MEASURES + MELODY_MEASURE_LENGTH
                           ) * N_BEATS * midi.ticks_per_beat
    last_note_end_tick = (INTRO_BLANK_MEASURES + MELODY_MEASURE_LENGTH +
                          1) * N_BEATS * midi.ticks_per_beat
    # 一番近い高低のbase noteを追加する
    last_transpose = round(melody_last_note / 12) * 12

    new_midi_paths = []
    for current_base_note in [key_base_note] + chord_I_notes:
        high_end_note = last_transpose + current_base_note
        low_end_note = last_transpose - (12 - current_base_note)
        for note_pitch in [high_end_note, low_end_note]:
            new_midi = copy.deepcopy(midi)
            new_midi.tracks[0].insert(
                -1,
                mido.Message('note_on', note=note_pitch, velocity=100, time=1))
            new_midi.tracks[0].insert(
                -1,
                mido.Message('note_off',
                             note=note_pitch,
                             velocity=0,
                             time=N_BEATS * midi.ticks_per_beat))
            new_midi_path = midi_file.with_stem(midi_file.stem +
                                                f'_end_{note_pitch}')
            new_midi.save(new_midi_path)
            new_midi_paths.append(new_midi_path)
    return new_midi_paths


def merge_backing_midi_and_melody_abc(input_melody_abc_path_or_str,
                                      input_backing_midi_path,
                                      output_melody_midi_path,
                                      output_full_midi_path,
                                      key,
                                      scale=None,
                                      replace_only_long=False):
    if not pathlib.Path(input_melody_abc_path_or_str).exists():
        abc_path = output_melody_midi_path.with_suffix('.abc')
        abc_path.write_text(input_melody_abc_path_or_str)
        input_melody_abc_path_or_str = abc_path
    backing_midi = mido.MidiFile(input_backing_midi_path)
    from_abc_to_midi(input_melody_abc_path_or_str,
                     output_melody_midi_path,
                     tempo=extract_bpm(backing_midi))

    if output_melody_midi_path.exists():
        # midiのkeyが信用できないため。
        # key = extract_key_signature(backing_midi)
        if scale:
            replace_note_with_scale_note(output_melody_midi_path,
                                         scale,
                                         key,
                                         replace_only_long=replace_only_long)

        melody_with_end_note_paths = append_end_note(output_melody_midi_path,
                                                     key)

        # TODO original の音程は使わないためmerge処理を省く
        for path in melody_with_end_note_paths:
            # 一旦2オクターブ上だけ
            for i in OCTAVES:
                # octaveあげたmidiをはく
                octave_melody_midi_path = path.with_stem(path.stem +
                                                         f'_{i}octave')

                generate_octave(path, octave_melody_midi_path, num_octave=i)

        # debug用
        try:
            song = music21.converter.parse(input_melody_abc_path_or_str)
            song.write('xml', output_melody_midi_path.with_suffix('.xml'))
        except Exception as e:
            logger.error(e)

        # song = music21.converter.parse(
        # song = music21.converter.parse(str(input_melody_abc_path))
        # song.write('xml', output_melody_midi_path.with_suffix('.xml'))
        # song = music21.converter.parse(
        #    str(output_melody_midi_path.with_suffix('.xml')))
        # song.insert(0, music21.tempo.MetronomeMark(number=120))

        # song.write('midi', output_melody_midi_path)

        # merge_backing_midi_and_melody_midi(
        #     input_backing_midi_path=input_backing_midi_path,
        #     output_melody_midi_path=output_melody_midi_path,
        #     output_full_midi_path=output_full_midi_path)

        for path in melody_with_end_note_paths:
            current_output_midi_path = path.with_stem(
                path.stem.replace('melody', 'full'))
            for i in OCTAVES:
                octave_full_midi_path = output_full_midi_path.with_stem(
                    current_output_midi_path.stem + f'_{i}octave')
                octave_melody_midi_path = path.with_stem(path.stem +
                                                         f'_{i}octave')
                merge_backing_midi_and_melody_midi(
                    input_backing_midi_path=input_backing_midi_path,
                    output_melody_midi_path=octave_melody_midi_path,
                    output_full_midi_path=octave_full_midi_path)


def merge_backing_midi_and_melody_midi(input_backing_midi_path,
                                       output_melody_midi_path,
                                       output_full_midi_path):
    melody_midi = mido.MidiFile(output_melody_midi_path)
    backing_midi = mido.MidiFile(input_backing_midi_path)
    new_tracks = []
    for message in melody_midi.tracks[0]:
        if not isinstance(
                message,
                mido.MetaMessage) and message.type != 'program_change':
            message.velocity = MELODY_VELOCITY
            new_tracks.append(message)
    melody_midi.tracks[0] = new_tracks
    backing_midi.tracks[1] = mido.merge_tracks(
        [backing_midi.tracks[1], melody_midi.tracks[0]])
    # backing_midi.tracks.append(melody_midi.tracks[0])
    backing_midi.save(output_full_midi_path)

    filepath = "your_midi_data.mid"
    # resolution: 四分音符ごとのタイムステップ数
    # Return type: ndarray, shape=(?, ?, 128)
    multitrack = pypianoroll.read(str(output_full_midi_path),
                                  resolution=MIDI_RESOLUTION)
    ax = multitrack.plot()
    plt.savefig(output_full_midi_path.with_suffix('.png'))
    plt.close()

    # synth = midi2audio.FluidSynth(
    #     sound_font='/usr/local/share/fluid-soundfont/FluidR3 GM2-2.SF2')
    # synth.midi_to_audio(output_full_midi_path,
    #                     output_melody_midi_path.with_suffix('.wav'))
    # パスが通らないことがあるため。遅い。
    # NOTE 遅いのでwavは生成しない。
    # subprocess.call([
    #     FLUIDSYNTH_PATH, '-ni',
    #     '/usr/local/share/fluid-soundfont/FluidR3 GM2-2.SF2',
    #     str(output_full_midi_path), '-F',
    #     str(output_full_midi_path.with_suffix('.wav')), '-r',
    #     str(midi2audio.DEFAULT_SAMPLE_RATE)
    # ])


def merge_all_backing_midi_and_melody_abcs(generated_abc_dir,
                                           backing_midi_path,
                                           generated_melody_midi_dir,
                                           generated_full_midi_dir):
    for path in generated_abc_dir.glob('*.abc'):
        logger.info('merging %s and %s', str(path), str(backing_midi_path))
        melody_midi_filename = generated_melody_midi_dir.joinpath(
            path.stem).with_suffix('.mid')
        output_full_midi_filename = generated_full_midi_dir.joinpath(
            path.stem).with_suffix('.mid')
        merge_backing_midi_and_melody_abc(
            input_melody_abc_path_or_str=path,
            input_backing_midi_path=backing_midi_path,
            output_melody_midi_path=melody_midi_filename,
            output_full_midi_path=output_full_midi_filename)


def from_bart_generation_to_abc_notation(response, song_name, model_name):
    header = f"X:1\nT:{song_name}\nC:${model_name}\nL:1/4\nM:4/4\nI:linebreak $\nK:C\nV:1\n"

    formatted_song = response.strip()
    formatted_song = formatted_song.replace('</s>', ' ')
    formatted_song = formatted_song.replace('`', '"')
    formatted_song = formatted_song.replace(" $ ", "\n")

    # 冒頭空白2小節 + intro2小節とendingの1小節を追加
    formatted_song = 'z4 | z4 | z4 | z4 |' + formatted_song + ' z4 |'
    return header + formatted_song


# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
from fairseq.models.bart import BARTModel
import argparse

XSUM_KWARGS = dict(beam=6,
                   lenpen=1.0,
                   max_len_b=60,
                   min_len=10,
                   no_repeat_ngram_size=3)
CNN_KWARGS = dict(beam=4,
                  lenpen=2.0,
                  max_len_b=140,
                  min_len=55,
                  no_repeat_ngram_size=3)


# TODO diverse beamなど実装する
@torch.no_grad()
def bart_generate(bart: BARTModel, source_xmls, **eval_kwargs):
    source_lines = []
    # FIXME メモリー上でもできるようにしておく
    for source_xml in source_xmls:
        abc_filename = source_xml.with_suffix('.abc')
        to_abc_notation(source_xml, abc_filename)
        source_line = from_abc_file_to_source_line(
            abc_notation_filename=abc_filename)
        source_lines.append(source_line)

    # if n_obs is not None: bsz = min(bsz, n_obs)
    #
    # with open(infile) as source, open(outfile, "w") as fout:
    #     sline = source.readline().strip()
    #     slines = [sline]
    #     for sline in source:
    #         if n_obs is not None and count > n_obs:
    #             break
    #         if count % bsz == 0:
    #             hypotheses_batch = bart.sample(slines, **eval_kwargs)
    #             for hypothesis in hypotheses_batch:
    #                 fout.write(hypothesis + "\n")
    #                 fout.flush()
    #             slines = []
    #
    #         slines.append(sline.strip())
    #         count += 1

    hypotheses_abcs = []
    for source_line, source_xml in zip(source_lines, source_xmls):
        src_dict = bart.task.source_dictionary
        tokens = src_dict.encode_line(
            source_line,
            add_if_not_exist=False,
            append_eos=True,
            reverse_order=False,
        ).long()
        hypotheses_batch = bart.generate(tokenized_sentences=[tokens],
                                         verbose=False,
                                         **eval_kwargs)[0]
        hypotheses_batch = [
            src_dict.string(tensor=hypothesis['tokens'],
                            bpe_symbol="sentencepiece",
                            escape_unk=True,
                            unk_string='*').replace(' ', '')
            for hypothesis in hypotheses_batch
        ]

        # hypotheses_batch = bart.sample(source_line, **eval_kwargs)

        hypothesis_abcs = []
        for hypothesis, xml_filename in zip(hypotheses_batch, source_xmls):
            # NOTE: 不正なスタイルが生成されることがあるため省く。
            hypothesis_abcs.append(
                from_bart_generation_to_abc_notation(hypothesis,
                                                     str(xml_filename),
                                                     'bart'))
        hypotheses_abcs.append(hypothesis_abcs)
    return hypotheses_abcs
    # for hypothesis in hypotheses_batch:
    #     fout.write(hypothesis + "\n")
    #     fout.flush()


def generate_melodies(model_dir, checkpoint_filename,
                      input_xml_dir: pathlib.Path, backing_midi_dir,
                      output_dir: pathlib.Path, generate_kwargs):
    """
    Usage::
         python examples/bart/summarize.py \
            --model-dir $HOME/bart.large.cnn \
            --model-file model.pt \
            --src $HOME/data-bin/cnn_dm/test.source
    """
    # parser = argparse.ArgumentParser()
    # parser.add_argument(
    #     "--model-dir",
    #     required=True,
    #     type=str,
    #     default="bart.large.cnn/",
    #     help="path containing model file and src_dict.txt",
    # )
    # parser.add_argument(
    #     "--model-file",
    #     default="checkpoint_best.pt",
    #     help="where in model_dir are weights saved",
    # )
    # parser.add_argument("--src",
    #                     default="test.source",
    #                     help="text to summarize",
    #                     type=str)
    # parser.add_argument("--out",
    #                     default="test.hypo",
    #                     help="where to save summaries",
    #                     type=str)
    # parser.add_argument("--bsz",
    #                     default=32,
    #                     help="where to save summaries",
    #                     type=int)
    # parser.add_argument("--n",
    #                     default=None,
    #                     help="how many examples to summarize",
    #                     type=int)
    # parser.add_argument(
    #     "--xsum-kwargs",
    #     action="store_true",
    #     default=False,
    #     help="if true use XSUM_KWARGS else CNN_KWARGS",
    # )
    # args = parser.parse_args()
    eval_kwargs = generate_kwargs

    bart = BARTModel.from_pretrained(model_dir,
                                     checkpoint_file=checkpoint_filename,
                                     data_name_or_path=model_dir,
                                     task='translation')
    bart = bart.eval()
    if torch.cuda.is_available():
        bart = bart.cuda().half()

    xml_paths = list(input_xml_dir.glob('**/*.xml'))
    backing_midi_paths = [
        backing_midi_dir.joinpath(
            path.relative_to(input_xml_dir)).parent.joinpath(
                path.name.replace('chord.xml', 'backing.mid'))
        for path in xml_paths
    ]
    # output_melody_midi_dir = output_dir.joinpath('melody_midi')
    # output_melody_midi_dir.mkdir(exist_ok=True, parents=True)
    #
    # output_melody_abc_dir = output_dir.joinpath('melody_abc')

    hypotheses = bart_generate(bart, xml_paths, **eval_kwargs)
    for i, (hypothesis_candidates,
            backing_midi_path) in enumerate(zip(hypotheses,
                                                backing_midi_paths)):
        current_output_root = output_dir.joinpath(
            backing_midi_path.relative_to(backing_midi_dir))
        for j, candidate in enumerate(hypothesis_candidates):
            current_candidate_root = current_output_root.joinpath(str(j))
            current_candidate_root.mkdir(exist_ok=True, parents=True)
            output_full_midi_path = current_candidate_root.joinpath('full.mid')
            output_melody_midi_path = current_candidate_root.joinpath(
                'melody.mid')
            output_melody_abc_path = current_candidate_root.joinpath(
                'melody.abc')
            # debugのため、開発中は一旦abcを保存しておく
            with output_melody_abc_path.open('w') as f:
                f.write(candidate)

            try:
                merge_backing_midi_and_melody_abc(
                    candidate,
                    backing_midi_path,
                    output_full_midi_path=output_full_midi_path,
                    output_melody_midi_path=output_melody_midi_path)
            except Exception as e:
                logger.error('failed to convert hypothesis %s',
                             str(current_candidate_root),
                             exc_info=e)


def from_all_chord_csvs_to_text2text_inputs(input_dir: pathlib.Path,
                                            output_dir: pathlib.Path,
                                            only_chord_change=False,
                                            fine_grained=False,
                                            original_xml2abc=False):
    samples = []
    for chord_path, key_file in zip(sorted(input_dir.glob('**/*chord.csv')),
                                    sorted(input_dir.glob('**/*key.txt'))):
        # melody_path = chord_path.parent.joinpath('melody.abc')
        # if melody_path.exists():
        key = key_file.read_text().split('\n')[0]
        key = 'C' if key == 'C major' else 'Amin'
        score = from_chord_csv_to_chord_score(
            chord_path,
            rest=True,
            chord_notes=True,
            only_chord_change=only_chord_change,
            fine_grained=fine_grained)
        score_path = output_dir.joinpath(str(
            chord_path.relative_to(input_dir))).with_suffix('.xml')

        score_path.parent.mkdir(exist_ok=True, parents=True)
        score.write('xml', score_path)
        abc_file = score_path.with_suffix('.abc')
        to_abc_notation(score_path, abc_file, original=original_xml2abc)

        source = from_abc_file_to_1line(abc_notation_filename=abc_file)

        samples.append({
            'chord': re.sub(r'K:\w+\n', f'K:{key}\n', source),
            'melody': 'dummy',
            'source_path': str(chord_path),
            'source_music': chord_path.parent.parent.name,
            'key': key
        })
    return samples
