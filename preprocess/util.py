import copy
import logging
import pathlib
import re
import subprocess
import sys
from collections.abc import Iterable

import music21
from numba import jit
from tqdm import tqdm

from xml2abc_143 import xml2abc

logger = logging.getLogger(__name__)
transpose_dict = {
    # major
    "G major": 5,
    "G# major": 4,
    "A- major": 4,
    "A major": 3,
    "A# major": 2,
    "B- major": 2,
    "B major": 1,
    "C major": 0,
    "C# major": -1,
    "D- major": -1,
    "D major": -2,
    "D# major": -3,
    "E- major": -3,
    "E major": -4,
    "F major": -5,
    "F# major": -6,
    "G- major": -6,
    # minor
    "E minor": 5,
    "F minor": 4,
    "F# minor": 3,
    "G- minor": 3,
    "G minor": 2,
    "G# minor": 1,
    "A- minor": 1,
    "A minor": 0,
    "A# minor": -1,
    "B- minor": -1,
    "B minor": -2,
    "C minor": -3,
    "C# minor": -4,
    "D- minor": -4,
    "D minor": -5,
    "D# minor": 6,
    "E- minor": 6
}


def remove_percussion_from_midi():
    percussion_promgrams = [0] + list(range(113, 129))
    removed_track_ids = []
    for i, track in enumerate(midi.tracks):
        for message in track:
            if message.type == 'program_change':
                if message.program in percussion_promgrams:
                    print(message.program)
                    removed_track_ids.append(i)
    for i in removed_track_ids:
        del midi.tracks[i]


def remove_percussion_part(score):
    new_score = copy.deepcopy(score)
    for part in new_score.parts:
        is_percussion_part = False
        for p in part:
            if isinstance(p, Iterable):
                for n in p:
                    if type(
                            n
                    ) == music21.percussion.PercussionChord or music21.note.Unpitched:
                        is_percussion_part = True
                        break
        if is_percussion_part:
            new_score.remove(part)
    return new_score


# @jit
def to_abc_notation(input_file, output_file, original=False):
    try:
        # originalと直接呼び出しで結果が微妙に変わるため切り替えられるようにしている。
        if original:
            output_bytes = subprocess.check_output([
                "python", "xml2abc_144_original/xml2abc.py",
                str(input_file), "-u", "-d", "4"
            ],
                                                   timeout=10)
            output = output_bytes.decode("utf-8").strip()
            with open(output_file, "w") as abc_file:
                abc_file.write(output)
        else:
            args = [str(input_file), "-u", "-d", "4"]

            xml2abc.main(args, output_file)

    except Exception as e:
        logger.error(e)


def transpose_and_save_xml_score(filename,
                                 output_filename,
                                 multipart=False,
                                 save_format='xml',
                                 only_four_beat=False,
                                 write_chord_notes=False):
    score = music21.converter.parse(filename)

    key = None
    metre = None
    part = score.parts[0]
    for p in part:
        if isinstance(p, Iterable):
            for n in p:
                if type(n) == music21.key.Key:
                    key = n.name
                if type(n) == music21.meter.TimeSignature:
                    metre = n.ratioString

    if key is None:
        try:
            key = score.analyze('key').name
        except AttributeError:
            percussion_removed_score = remove_percussion_part(score)
            key = percussion_removed_score.analyze('key').name

    # 小文字のケースがあるため
    key = key[:1].upper() + key[1:]
    print(key, metre)

    if only_four_beat and not metre == "4/4":
        return

    if key in transpose_dict.keys():
        interval = transpose_dict[key]
        print("transposing from key", key, "to C major using interval",
              interval)
        try:
            score = score.transpose(interval)
        # except music21.midi.MidiException as e:
        #     print(e, 'here')
        #     raise e
        except Exception as e:
            print(e)
            print('percussion parts are removing')
            score = remove_percussion_part(score).transpose(interval)
    else:
        print("unknown key", key, "skipped")

    # part = score.parts[0]
    # inversionがxml読み込み時にエラーになることがあるため、消しておく
    for part in score.parts:
        for p in part:
            if isinstance(p, Iterable):
                for n in p:
                    if type(n) == music21.harmony.ChordSymbol:
                        if not n.inversionIsValid(n.inversion()):
                            n.inversion(0)
                        if write_chord_notes:
                            n.writeAsChord = write_chord_notes

                    if isinstance(n, music21.key.KeySignature):
                        p.replace(n, music21.key.KeySignature(0))

    score.write(save_format, fp=output_filename)


def from_all_xml_files_to_transposed_xml(
        raw_data_root,
        score_data_root,
        require_abc=False,
        abc_notation_root: pathlib.Path = None):
    for s in list(raw_data_root.glob('**/*.xml')) + list(
            raw_data_root.glob('**/*.mxl')):
        xml_filename = score_data_root.joinpath(
            pathlib.Path(s).relative_to(raw_data_root).with_suffix('.xml'))
        xml_filename.parent.mkdir(exist_ok=True, parents=True)
        transpose_and_save_xml_score(s, xml_filename)

        if require_abc and xml_filename.exists():
            abc_filename = abc_notation_root.joinpath(
                pathlib.Path(s).relative_to(raw_data_root).with_suffix('.txt'))
            abc_filename.parent.mkdir(exist_ok=True, parents=True)
            to_abc_notation(xml_filename, abc_filename)


def extract_melody_score(score):
    new_score = copy.deepcopy(score)
    for p in new_score.parts[0]:
        if isinstance(p, music21.spanner.Spanner) and not isinstance(
                p, music21.spanner.Slur):
            p.show('text')
            if isinstance(p, music21.spanner.RepeatBracket):
                new_score = strip_repeat_bracket(p, new_score)
            elif isinstance(p, music21.spanner.MultiMeasureRest):
                continue
            else:
                raise ValueError('not supported spanner')

    for p in new_score.parts[0]:
        if isinstance(p, Iterable) and not isinstance(
                p, music21.spanner.MultiMeasureRest):
            for n in p:
                # p.show('text')
                if isinstance(n, music21.harmony.ChordSymbol):
                    p.remove(n)
                if isinstance(n, music21.bar.Repeat):
                    p.replace(n, music21.bar.Barline(location=n.location))
    return new_score


def strip_slur(slur_part, score):
    # spannerの場合はreplaceができないためpartに変換する。
    print('slur is removing')
    new_part = music21.stream.Part()
    new_part.offset = slur_part.offset
    # new_part.id = slur_part.id
    for n in slur_part:
        new_part.append(n)
    score.parts[0].replace(slur_part, new_part)
    return score


def strip_repeat_bracket(part, score):
    score.parts[0].remove(part)
    return score


# In[321]:


# @jit
def extract_chord_score(score, rest=False, explicit_chord_cross_bar=False):
    new_score = copy.deepcopy(score)
    for p in new_score.parts[0]:
        if isinstance(p, music21.spanner.Spanner):
            # p.show('text')
            if isinstance(p, music21.spanner.Slur):
                new_score = strip_slur(p, new_score)
            elif isinstance(p, music21.spanner.RepeatBracket):
                new_score = strip_repeat_bracket(p, new_score)
            elif isinstance(p, music21.spanner.MultiMeasureRest):
                continue
            else:
                raise ValueError('not supported spanner')
    # new_parts = [strip_slur(p) for p in new_score.parts[0]]

    for p in new_score.parts[0]:
        if isinstance(p, Iterable):
            for n in p:
                if isinstance(n, music21.note.Note):
                    p.replace(n, music21.note.Rest(n.duration.quarterLength))
                if isinstance(n, music21.bar.Repeat):
                    p.replace(n, music21.bar.Barline(location=n.location))

    merge_rest(new_score)
    if not rest:
        replace_rest_to_chord_notes(
            new_score.parts[0],
            explicit_chord_cross_bar=explicit_chord_cross_bar)
    else:
        add_chord_notes(new_score.parts[0])

    return new_score


def merge_rest(new_score):
    for p in new_score.parts[0]:
        # 基本的にnoteの前にコードが先にくる。
        if isinstance(p, Iterable) and not isinstance(
                p, music21.spanner.MultiMeasureRest):
            current_rest = None
            current_harmony_rests = []
            for n in p:
                # n.show('text')
                # print(n.offset)
                # print(current_harmony_rest)

                if isinstance(n, music21.harmony.ChordSymbol) or isinstance(
                        n, music21.bar.Barline):
                    # current_rest = None
                    # print(current_harmony_rests)
                    # print(len(current_harmony_rests) >= 2)
                    if len(current_harmony_rests) >= 2:
                        print('merging ', str(current_harmony_rests))
                        new_rest = music21.note.Rest(
                            sum([
                                rest.duration.quarterLength
                                for rest in current_harmony_rests
                            ]))
                        new_rest.offset = current_harmony_rests[0].offset
                        # print('new rest ', str(new_rest.show('text')))
                        p.replace(current_harmony_rests[0], new_rest)
                        for old_rest in current_harmony_rests[1:]:
                            p.remove(old_rest)
                    current_harmony_rests = []

                elif isinstance(n, music21.note.Rest):
                    # if current_rest:
                    #     current_rest.mergeWith(n)
                    # else:
                    #     current_rest = n
                    current_harmony_rests.append(n)

            if len(current_harmony_rests) >= 2:
                # print('merging ', str(current_harmony_rest))
                new_rest = music21.note.Rest(
                    sum([
                        rest.duration.quarterLength
                        for rest in current_harmony_rests
                    ]))
                new_rest.offset = current_harmony_rests[0].offset
                # print('new rest ', str(new_rest.show('text')))
                p.replace(current_harmony_rests[0], new_rest)
                for old_rest in current_harmony_rests[1:]:
                    p.remove(old_rest)
            current_harmony_rests = []


def add_chord_notes(part):
    for p in part:
        if isinstance(p, Iterable):
            for n in p:
                if isinstance(n, music21.harmony.ChordSymbol):
                    n.writeAsChord = True


def make_explicit_chord(part):
    """
    小節を跨ぐ場合に明治的にchordsymbolを入れる。
    :param p:
    :return:
    """
    current_chord = None
    cross_bar = False
    continuous_rest = False
    # ２小節連続で休符が続く場合は対応しない。
    for p in part:
        if isinstance(p, Iterable):
            replacer = []
            for i, n in enumerate(p):
                if isinstance(n, music21.harmony.ChordSymbol):
                    cross_bar = False
                    current_chord = n
                elif current_chord and cross_bar and isinstance(
                        n, music21.note.Rest):
                    inserted_chord = music21.harmony.ChordSymbol(
                        # root=current_chord.root(),
                        # bass=current_chord.bass(),
                        kindStr=current_chord.chordKindStr,
                        inversion=current_chord.inversion(),
                        kind=current_chord.chordKind,
                        notes=current_chord.notes,
                        pitches=current_chord.pitches,
                    )
                    inserted_chord.offset = n.offset
                    replacer.append(inserted_chord)
                    # cross_bar = False
                    # current_chord = None

            for chord in replacer:
                p.insert(chord)
        cross_bar = True


def replace_rest_to_chord_notes(part, explicit_chord_cross_bar=False):
    if explicit_chord_cross_bar:
        make_explicit_chord(part)
    for p in part:
        if isinstance(p, Iterable):
            previous_is_chord = False
            current_chord = None
            for n in p:
                # n.show('text')
                # print(n.offset)
                # print(current_harmony_rest)
                if isinstance(
                        n,
                        music21.harmony.ChordSymbol) and not previous_is_chord:
                    n.writeAsChord = True
                    current_chord = n
                    previous_is_chord = True
                elif isinstance(
                        n, music21.harmony.ChordSymbol) and previous_is_chord:
                    p.remove(n)
                elif isinstance(n, music21.note.Rest) and current_chord:
                    current_chord.quarterLength = copy.deepcopy(
                        n.quarterLength)
                    p.remove(n)
                    previous_is_chord = False


def remove_final_barline(score):
    for p in score.parts[0]:
        if isinstance(p, Iterable):
            for n in p:
                if isinstance(n, music21.bar.Barline):
                    p.remove(n)
    return score


def segment_all_melody_and_chords(melody_score_root: pathlib.Path,
                                  chord_root_dir: pathlib.Path,
                                  segment_root_dir: pathlib.Path,
                                  melody_window,
                                  chord_window,
                                  stride,
                                  use_cache=False):
    """
    relative pathが同じ前提
    :param melody_score_root:
    :param chord_root_dir:
    :param segment_root_dir:
    :return:
    """
    melody_paths = sorted(melody_score_root.glob('**/*.xml'))
    # chords = sorted(chord_root_dir.glob('**/*.xml'))

    for melody_score_path in tqdm(melody_paths):
        logger.info('processing %s', str(melody_score_path))
        chord_score_path = chord_root_dir.joinpath(
            melody_score_path.relative_to(melody_score_root))

        if not chord_score_path.exists():
            logger.warning('chord for %s not exists', str(melody_score_path))
            continue

        output_dir = segment_root_dir.joinpath(
            melody_score_path.relative_to(melody_score_root).with_suffix(''))
        if use_cache and output_dir.exists():
            logger.info('segment for %s exist', str(melody_score_path))
            continue

        output_dir.mkdir(exist_ok=True, parents=True)

        segment_melody_and_chord(melody_score_path,
                                 chord_score_path,
                                 output_dir,
                                 melody_window=melody_window,
                                 chord_window=chord_window,
                                 stride=stride)


def segment_melody_and_chord(melody_score_path,
                             chord_path,
                             segment_music_dir,
                             melody_window=8,
                             chord_window=10,
                             stride=2):
    print(melody_score_path)
    assert melody_score_path.stem == chord_path.stem
    melody_score = music21.converter.parse(melody_score_path)
    chord_score = music21.converter.parse(chord_path)

    n_measures = len(
        melody_score.getElementsByClass(
            music21.stream.Part)[0].getElementsByClass(music21.stream.Measure))

    start_index = 1
    # 奇数の場合小説の途中から始まっている可能性があるため
    if n_measures % 2:
        start_index = 2
    indices = [
        ((i, i + chord_window - 1), (i + chord_window - melody_window,
                                     i + chord_window - 1))
        for i in range(start_index, n_measures - chord_window + stride, stride)
    ]

    # inclusive
    melody_segment_scores = [
        remove_final_barline(melody_score.measures(start, end))
        for _, (start, end) in indices
    ]
    chord_segment_scores = [
        remove_final_barline(chord_score.measures(start, end))
        for (start, end), _ in indices
    ]

    segment_music_dir.mkdir(exist_ok=True, parents=True)
    for i, (melody_segment_score, chord_segment_score) in enumerate(
            zip(melody_segment_scores, chord_segment_scores)):
        segment_dir = segment_music_dir.joinpath(f'{i:08}')
        segment_dir.mkdir(exist_ok=True, parents=True)
        try:
            melody_segment_score.write('xml',
                                       segment_dir.joinpath('melody.xml'))
            chord_segment_score.write('xml', segment_dir.joinpath('chord.xml'))
        except Exception as e:
            logger.error(e)


def from_full_score_to_chord_and_melody(score_data_root,
                                        chord_root_dir,
                                        melody_score_root,
                                        chord_notes=False,
                                        explicit_chord_cross_bar=False,
                                        only_single_part=False):
    for path in tqdm(score_data_root.glob('**/*.xml')):
        try:
            print(path)
            score = music21.converter.parse(path)
            if only_single_part and len(score.parts) >= 2:
                logger.warning('%s has more than one part, skipped', str(path))
                continue
            chord_score = extract_chord_score(
                score,
                rest=chord_notes,
                explicit_chord_cross_bar=explicit_chord_cross_bar)
            chord_score_path = chord_root_dir.joinpath(
                path.relative_to(score_data_root))
            chord_score_path.parent.mkdir(exist_ok=True, parents=True)
            chord_score.write('xml', fp=chord_score_path)

            melody_score = extract_melody_score(score)
            melody_score_path = melody_score_root.joinpath(
                path.relative_to(score_data_root))
            melody_score_path.parent.mkdir(exist_ok=True, parents=True)
            melody_score.write('xml', fp=melody_score_path)
        except Exception as error:
            logging.error('failed to process %s', str(path), exc_info=error)


def extract_single_note_from_multiple_notes(abc_notation):
    return re.sub(r'\[([a-zA-Z][,\']*)([a-zA-Z][,\']*)*\]', r'\1',
                  abc_notation)
    # return re.sub(r'\[([a-zA-z][,\']*)\w+\]', '\1', note)
