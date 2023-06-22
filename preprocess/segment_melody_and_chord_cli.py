"""
スクリプト
"""
import pathlib

from absl import app
from absl import flags
import logging
from preprocess import util

FLAGS = flags.FLAGS
flags.DEFINE_string('melody_root_dir', '', '入力')
flags.DEFINE_string('chord_root_dir', '', '入力')

flags.DEFINE_string('output_dir', '', '出力')
flags.DEFINE_integer('melody_window', 8, '')
flags.DEFINE_integer('chord_window', 8, '')
flags.DEFINE_integer('stride', 4, '')
flags.DEFINE_bool('use_cache', False, '')

logger = logging.getLogger(__name__)


def main(unused_argv) -> None:
    """main関数"""
    util.segment_all_melody_and_chords(pathlib.Path(FLAGS.melody_root_dir),
                                       pathlib.Path(FLAGS.chord_root_dir),
                                       pathlib.Path(FLAGS.output_dir),
                                       melody_window=FLAGS.melody_window,
                                       chord_window=FLAGS.chord_window,
                                       stride=FLAGS.stride,
                                       use_cache=FLAGS.use_cache)


if __name__ == '__main__':
    app.run(main)
