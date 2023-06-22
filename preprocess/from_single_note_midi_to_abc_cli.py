"""
スクリプト
"""
import os
import pathlib
import subprocess

from absl import app
from absl import flags
import logging

FLAGS = flags.FLAGS
flags.DEFINE_string('input_dir', '', '入力')
flags.DEFINE_string('output_dir', '', '出力')

logger = logging.getLogger(__name__)


def main(unused_argv) -> None:
    """main関数"""
    input_dir_path = pathlib.Path(FLAGS.input_dir)
    output_dir_path = pathlib.Path(FLAGS.output_dir)
    output_dir_path.mkdir(exist_ok=True, parents=True)

    for midi_path in input_dir_path.glob('**/*.mid'):
        logger.info('converting %s to abc', str(midi_path))
        output_abc_path = output_dir_path.joinpath(
            midi_path.relative_to(input_dir_path)).with_suffix('.abc')
        output_bytes = subprocess.check_output(
            ['midi2abc',
             str(midi_path), "-v", "-o",
             str(output_abc_path)],
            timeout=20)
        print(output_bytes)


if __name__ == '__main__':
    app.run(main)
