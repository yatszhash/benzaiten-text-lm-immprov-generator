import copy
import logging

import mido
import music21
import pandas as pd

from preprocess.util import merge_rest, replace_rest_to_chord_notes

TOTAL_MEASURES = 240  # 学習用MusicXMLを読み込む際の小節数の上限
UNIT_MEASURES = 4  # 1回の生成で扱う旋律の長さ
BEAT_RESO = 4  # 1拍を何個に分割するか（4の場合は16分音符単位）
N_BEATS = 4  # 1小節の拍数（今回は4/4なので常に4）
NOTENUM_FROM = 36  # 扱う音域の下限（この値を含む）
NOTENUM_THRU = 84  # 扱う音域の上限（この値を含まない）
INTRO_BLANK_MEASURES = 4  # ブランクおよび伴奏の小節数の合計
MELODY_LENGTH = 8  # 生成するメロディの長さ（小節数）
MIDI_RESOLUTION = 480

KEY_ROOT = "C"  # 生成するメロディの調のルート（"C" or "A"）
KEY_MODE = "major"  # 生成するメロディの調のモード（"major" or "minor"）

# defaultが1024で合わないので、書き換えておく
music21.defaults.ticksPerQuarter = MIDI_RESOLUTION
logger = logging.getLogger(__name__)


# 指定された仕様のcsvファイルを読み込んで
# ChordSymbol列を返す
def to_chord_score(chord_df,
                   rest=True,
                   chord_notes=False,
                   only_chord_change=False,
                   fine_grained=False):
    # TODO chordの変換表を作る。
    # >> > symbols = ['', 'm', '+', 'dim', '7',
    #                 ...            'M7', 'm7', 'dim7', '7+', 'm7b5',  # half-diminished
    #                 ...            'mM7', '6', 'm6', '9', 'Maj9', 'm9',
    #                 ...            '11', 'Maj11', 'm11', '13',
    #                 ...            'Maj13', 'm13', 'sus2', 'sus4',
    #                 ...            'N6', 'It+6', 'Fr+6', 'Gr+6', 'pedal',
    #                 ...            'power', 'tristan', '/E', 'm7/E-', 'add2',
    #                 ...            '7omit3', ]
    score = music21.stream.Score()
    part = music21.stream.Part()
    current_measure = music21.stream.Measure(number=1, offset=0.0)
    current_measure.append(music21.meter.TimeSignature('4/4'))
    part.insert(current_measure)
    if chord_df.iloc[0]['root'].lower() == 'a':
        key = music21.key.Key('Am')
    else:
        key = music21.key.Key('C')
    current_measure.append(key)
    current_measure.append(music21.key.KeySignature(key.sharps))

    previous_chord = None
    for i, row in chord_df.iterrows():
        # 8小節を超えたら無視する。
        if row['measure'] + 1 >= 9:
            break

        if current_measure.number != row['measure'] + 1:
            current_measure = music21.stream.Measure(number=row['measure'] + 1,
                                                     offset=row['measure'] * 4)
            part.insert(current_measure)
        current_chord = music21.harmony.ChordSymbol(root=row['root'],
                                                    kind=row['kind'],
                                                    bass=row['base'])
        # train時のmusic xmlに合わせてchordが変わらない場合追加しない。
        if previous_chord is None or (
                previous_chord.notes != current_chord.notes
                or not only_chord_change):
            if chord_notes:
                current_chord.writeAsChord = True
            current_measure.insert(row['beat'], current_chord)
            previous_chord = current_chord

        if i < chord_df.shape[0] - 1:
            if fine_grained:
                for i in range(2):
                    # FIXME フローが汚い
                    if i >= 1:
                        current_measure.insert(row['beat'] + i,
                                               copy.deepcopy(current_chord))
                    current_measure.insert(row['beat'] + i,
                                           music21.note.Rest())

            else:
                current_measure.insert(row['beat'], music21.note.Rest('half'))
        else:
            if fine_grained:
                for _ in range(4):
                    if i >= 1:
                        current_measure.insert(row['beat'] + i,
                                               copy.deepcopy(current_chord))
                    current_measure.insert(row['beat'], music21.note.Rest())
            else:
                current_measure.insert(row['beat'], music21.note.Rest('whole'))

    score.insert(part)
    if only_chord_change:
        merge_rest(score)

    if rest:
        replace_rest_to_chord_notes(part)

    return score


def from_chord_csv_to_chord_score(chord_csv,
                                  rest=True,
                                  chord_notes=False,
                                  only_chord_change=False,
                                  fine_grained=False):
    chords = pd.read_csv(chord_csv,
                         names=['measure', 'beat', 'root', 'kind', 'base'])
    return to_chord_score(chords,
                          rest,
                          chord_notes,
                          only_chord_change=only_chord_change,
                          fine_grained=fine_grained)


def merge_backing_midi_and_melody_abc(input_melody_abc_path,
                                      input_backing_midi_path,
                                      output_melody_midi_path,
                                      output_midi_path):
    song = music21.converter.parse(str(input_melody_abc_path))
    song.write('midi', output_melody_midi_path)
    merge_backing_midi_and_melody_midi(
        input_backing_midi_path=input_backing_midi_path,
        output_melody_midi_path=output_melody_midi_path,
        output_full_midi_path=output_midi_path)


def merge_backing_midi_and_melody_midi(input_backing_midi_path,
                                       output_melody_midi_path,
                                       output_full_midi_path):
    melody_midi = mido.MidiFile(output_melody_midi_path)
    backing_midi = mido.MidiFile(input_backing_midi_path)
    backing_midi.tracks[1] = mido.merge_tracks(
        [backing_midi.tracks[1], melody_midi.tracks[1]])
    backing_midi.save(output_full_midi_path)


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
            input_melody_abc_path=path,
            input_backing_midi_path=backing_midi_path,
            output_melody_midi_path=melody_midi_filename,
            output_midi_path=output_full_midi_filename)


def from_gpt3_response_to_abc_notation(response, song_name, model_name):
    header = f"X:1\nT:{song_name}\nC:${model_name}\nL:1/4\nM:4/4\nI:linebreak $\nK:C\nV:1\n"

    formatted_song = response["choices"][0]["text"].strip()
    formatted_song = formatted_song.replace('`', '"')
    formatted_song = formatted_song.replace(" $ ", "\n")

    # 冒頭空白2小節 + intro2小節とendingの1小節を追加
    formatted_song = 'z4 | z4 | z4 | z4 |' + formatted_song + ' z4 |'
    return header + formatted_song
