import pathlib
from absl import app
from absl import flags
from numba import jit

from preprocess.util import from_full_score_to_chord_and_melody

FLAGS = flags.FLAGS
flags.DEFINE_string('input', '', '入力')
flags.DEFINE_string('melody_root_dir', '', '出力')
flags.DEFINE_string('chord_root_dir', '', '出力')
flags.DEFINE_bool('chord_notes', False, '休符を残しつつ、chord_noteを置き換えるか？')
flags.DEFINE_bool('explicit_chord_cross_bar', False, '')
flags.DEFINE_bool('only_single_part', False, '')


def main(unused_argv):
    # score_data_root = pathlib.Path('../data/OpenEWLD/v002/dataset')
    chord_root_dir = pathlib.Path(FLAGS.chord_root_dir)
    melody_root_dir = pathlib.Path(FLAGS.melody_root_dir)
    # from_full_score_to_chord_and_melody(score_data_root, chord_root_dir,
    #                                     melody_root_dir)
    from_full_score_to_chord_and_melody(
        pathlib.Path(FLAGS.input),
        chord_root_dir,
        melody_root_dir,
        chord_notes=FLAGS.chord_notes,
        explicit_chord_cross_bar=FLAGS.explicit_chord_cross_bar,
        only_single_part=FLAGS.only_single_part)


if __name__ == '__main__':
    app.run(main)
