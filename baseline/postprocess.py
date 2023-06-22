import csv

import mido
import music21
from pretty_midi import pretty_midi
import numpy as np

from gpt3.inference import BEAT_RESO, INTRO_BLANK_MEASURES, MELODY_LENGTH, N_BEATS, NOTENUM_FROM, \
    NOTENUM_THRU


def make_empty_pianoroll(length):
    return np.zeros((length, NOTENUM_THRU - NOTENUM_FROM + 1))


# ピアノロールを描画し、MIDIファイルを再生
def genereate_full_midi(pianoroll, transpose, src_filename, dst_filename):
    # ピアノロール（one-hot vector列）をノートナンバー列に変換
    def calc_notenums_from_pianoroll(pianoroll):
        notenums = []
        for i in range(pianoroll.shape[0]):
            n = np.argmax(pianoroll[i, :])
            nn = -1 if n == pianoroll.shape[1] - 1 else n + NOTENUM_FROM
            notenums.append(nn)
        return notenums

    # 連続するノートナンバーを統合して (notenums, durations) に変換
    def calc_durations(notenums):
        N = len(notenums)
        duration = [1] * N
        for i in range(N):
            k = 1
            while i + k < N:
                if notenums[i] > 0 and notenums[i] == notenums[i + k]:
                    notenums[i + k] = 0
                    duration[i] += 1
                else:
                    break
                k += 1
        return notenums, duration

    # MIDIファイルを生成
    def make_midi(notenums, durations, transpose, src_filename, dst_filename):
        midi = mido.MidiFile(src_filename)
        MIDI_DIVISION = midi.ticks_per_beat
        track = mido.MidiTrack()
        midi.tracks.append(track)
        init_tick = INTRO_BLANK_MEASURES * N_BEATS * MIDI_DIVISION
        prev_tick = 0
        for i in range(len(notenums)):
            if notenums[i] > 0:
                curr_tick = int(i * MIDI_DIVISION / BEAT_RESO) + init_tick
                track.append(
                    mido.Message('note_on',
                                 note=notenums[i] + transpose,
                                 velocity=100,
                                 time=curr_tick - prev_tick))
                prev_tick = curr_tick
                curr_tick = int(
                    (i + durations[i]) * MIDI_DIVISION / BEAT_RESO) + init_tick
                track.append(
                    mido.Message('note_off',
                                 note=notenums[i] + transpose,
                                 velocity=100,
                                 time=curr_tick - prev_tick))
                prev_tick = curr_tick
        midi.save(dst_filename)

    # plt.matshow(np.transpose(pianoroll))
    # plt.show()
    notenums = calc_notenums_from_pianoroll(pianoroll)
    notenums, durations = calc_durations(notenums)
    make_midi(notenums, durations, transpose, src_filename, dst_filename)
    # fs = midi2audio.FluidSynth(sound_font="/usr/share/sounds/sf2/FluidR3_GM.sf2")
    # fs.midi_to_audio(dst_filename, "output.wav")
    # ipd.display(ipd.Audio("output.wav"))


# 指定された仕様のcsvファイルを読み込んで
# ChordSymbol列を返す
def read_chord_file(file):
    chord_seq = [None] * (MELODY_LENGTH * N_BEATS)
    with open(file) as f:
        reader = csv.reader(f)
        for row in reader:
            m = int(row[0])  # 小節番号（0始まり）
            if m < MELODY_LENGTH:
                b = int(row[1])  # 拍番号（0始まり、今回は0または2）
                chord_seq[m * 4 + b] = music21.harmony.ChordSymbol(root=row[2],
                                                                   kind=row[3],
                                                                   bass=row[4])
    for i in range(len(chord_seq)):
        if chord_seq[i] != None:
            chord = chord_seq[i]
        else:
            chord_seq[i] = chord
    return chord_seq


# ChordSymbol列をmany-hot (chroma) vector列に変換
def chord_seq_to_chroma(chord_seq):
    N = len(chord_seq)
    matrix = np.zeros((N, 12))
    for i in range(N):
        if chord_seq[i] != None:
            for note in chord_seq[i]._notes:
                matrix[i, note.pitch.midi % 12] = 1
    return matrix


# コード進行からChordSymbol列を生成
# divisionは1小節に何個コードを入れるか
def make_chord_seq(chord_prog, division):
    T = int(N_BEATS * BEAT_RESO / division)
    seq = [None] * (T * len(chord_prog))
    for i in range(len(chord_prog)):
        for t in range(T):
            if isinstance(chord_prog[i], music21.harmony.ChordSymbol):
                seq[i * T + t] = chord_prog[i]
            else:
                seq[i * T + t] = music21.harmony.ChordSymbol(chord_prog[i])
    return seq


def merge_backing_midi_and_melody_midi(chord_filepath, backing_filepath,
                                       melody_filepath, output_filepath):
    chord_prog = read_chord_file(chord_filepath)
    chroma_vec = chord_seq_to_chroma(make_chord_seq(chord_prog, N_BEATS))
    pianoroll = make_empty_pianoroll(chroma_vec.shape[0])
    midi_data = pretty_midi.PrettyMIDI(melody_filepath)

    for instrument in midi_data.instruments:
        for note in instrument.notes:
            start, end = int(note.start * MELODY_LENGTH), int(
                note.end * MELODY_LENGTH)  # ループ内で見ているnoteの開始点と終了点を算出
            pianoroll[start:end, :] = np.zeros(NOTENUM_THRU - NOTENUM_FROM +
                                               1, )
            pianoroll[start:end, note.pitch -
                                 NOTENUM_FROM] = 1  # note.pitchのところのみ1とするone-hotベクトルを書き込む

    genereate_full_midi(pianoroll, 12, backing_filepath, output_filepath)
