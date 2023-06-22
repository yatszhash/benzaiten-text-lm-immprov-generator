"""
スクリプト
tokenizeまで入っている。
"""
import logging
import pathlib

from absl import app
from absl import flags

from bart.inference import from_all_chord_csvs_to_text2text_inputs
import pandas as pd

FLAGS = flags.FLAGS
flags.DEFINE_string('input_dir', '', '入力')
flags.DEFINE_string('output_filename', '', '出力')
# flags.DEFINE_float('valid_ratio', 0.1, '')
# flags.DEFINE_string('output_reference_filename', '', 'referenceの出力先')
# flags.DEFINE_enum('task', 'pretrain', ['pretrain', 'generate'], '')

logger = logging.getLogger(__name__)


def main(unused_argv) -> None:
    """main関数"""
    input_dir = pathlib.Path(FLAGS.input_dir)

    output_filename = pathlib.Path(FLAGS.output_filename)
    output_filename.parent.mkdir(exist_ok=True, parents=True)
    output_dir = output_filename.parent
    # if FLAGS.task == 'pretrain':
    #     samples = []
    #     for path in input_dir.glob('**/*.abc'):
    #         prompt = from_abc_file_to_source_line(path)
    #         samples.append(prompt)
    #     with output_filename.open('w') as fp:
    #         fp.write('\n'.join(samples) + '\n')
    # else:
    samples = from_all_chord_csvs_to_text2text_inputs(input_dir, output_dir)
    df = pd.DataFrame.from_records(samples)

    df.to_json(output_filename,
               orient='records',
               force_ascii=False,
               lines=True)

    # train, valid = train_test_split(df,
    #                                 test_size=FLAGS.valid_ratio,
    #                                 random_state=0,
    #                                 stratify=df['source_music'])
    # train.to_json(output_filename.parent.joinpath('train_jsonl.json'),
    #               orient='records',
    #               force_ascii=False,
    #               lines=True)
    # train.to_json(output_filename.parent.joinpath('valid_jsonl.json'),
    #               orient='records',
    #               force_ascii=False,
    #               lines=True)
    # with output_filename.open('w') as fp:
    #     writer = jsonlines.Writer(fp)
    #     writer.write_all(samples)
    #     writer.close()


if __name__ == '__main__':
    app.run(main)
