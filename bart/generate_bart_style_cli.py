"""
スクリプト
tokenizeまで入っている。
"""
import logging
import pathlib

from absl import app
from absl import flags
from jsonlines import jsonlines

from gpt3.generate_prompt import from_abc_file_to_source_line
import pandas as pd

FLAGS = flags.FLAGS
flags.DEFINE_string('input_dir', '', '入力')
flags.DEFINE_string('output_filename', '', '出力')
flags.DEFINE_string('output_reference_filename', '', 'referenceの出力先')
flags.DEFINE_enum('task', 'pretrain', ['pretrain', 'generate'], '')

logger = logging.getLogger(__name__)


def main(unused_argv) -> None:
    """main関数"""
    input_dir = pathlib.Path(FLAGS.input_dir)

    output_filename = pathlib.Path(FLAGS.output_filename)
    output_filename.parent.mkdir(exist_ok=True, parents=True)
    if FLAGS.task == 'pretrain':
        samples = []
        for path in input_dir.glob('**/*.abc'):
            prompt = from_abc_file_to_source_line(path)
            samples.append(prompt)
        with output_filename.open('w') as fp:
            fp.write('\n'.join(samples) + '\n')
    else:
        sources = []
        references = []
        output_reference_filename = pathlib.Path(
            FLAGS.output_reference_filename)
        for chord_path in input_dir.glob('**/chord.abc'):
            melody_path = chord_path.parent.joinpath('melody.abc')
            if melody_path.exists():
                source = from_abc_file_to_source_line(chord_path)
                reference = from_abc_file_to_source_line(melody_path)
                sources.append(source)
                references.append(reference)
        with output_filename.open('w') as fp:
            fp.write('\n'.join(sources) + '\n')

        with output_reference_filename.open('w') as fp:
            fp.write('\n'.join(references) + '\n')

    # with output_filename.open('w') as fp:
    #     writer = jsonlines.Writer(fp)
    #     writer.write_all(samples)
    #     writer.close()


if __name__ == '__main__':
    app.run(main)
