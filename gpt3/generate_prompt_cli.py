"""
スクリプト
"""
import pathlib

from absl import app
from absl import flags
import logging

from gpt3.generate_prompt import generate_to_prompt_from_abc_files

FLAGS = flags.FLAGS
flags.DEFINE_string('input', '', '入力')
flags.DEFINE_string('output', '', '出力')

logger = logging.getLogger(__name__)


def main(unused_argv) -> None:
    """main関数"""
    prompt = generate_to_prompt_from_abc_files(pathlib.Path(FLAGS.input))
    output_filename = pathlib.Path(FLAGS.output)
    output_filename.parent.mkdir(exist_ok=True, parents=True)
    output_filename.open('w').write(''.join(prompt))


if __name__ == '__main__':
    app.run(main)
