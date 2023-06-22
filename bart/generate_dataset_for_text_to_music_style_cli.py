"""
スクリプト
tokenizeまで入っている。
"""
import logging
import pathlib

from absl import app
from absl import flags
from jsonlines import jsonlines

from gpt3.generate_prompt import from_abc_file_to_1line, from_abc_file_to_source_line
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split

FLAGS = flags.FLAGS
flags.DEFINE_string('input_dir', '', '入力')
flags.DEFINE_string('output_filename', '', '出力')
flags.DEFINE_float('valid_ratio', 0.1, '')
flags.DEFINE_bool('source_prefix', False, '')
# flags.DEFINE_string('output_reference_filename', '', 'referenceの出力先')
# flags.DEFINE_enum('task', 'pretrain', ['pretrain', 'generate'], '')

logger = logging.getLogger(__name__)


def main(unused_argv) -> None:
    """main関数"""
    input_dir = pathlib.Path(FLAGS.input_dir)

    output_filename = pathlib.Path(FLAGS.output_filename)
    output_filename.parent.mkdir(exist_ok=True, parents=True)
    # if FLAGS.task == 'pretrain':
    #     samples = []
    #     for path in input_dir.glob('**/*.abc'):
    #         prompt = from_abc_file_to_source_line(path)
    #         samples.append(prompt)
    #     with output_filename.open('w') as fp:
    #         fp.write('\n'.join(samples) + '\n')
    # else:
    samples = []
    output_reference_filename = pathlib.Path(FLAGS.output_filename)
    for chord_path in input_dir.glob('**/chord.abc'):
        melody_path = chord_path.parent.joinpath('melody.abc')
        if melody_path.exists():
            source = from_abc_file_to_1line(chord_path)
            reference = from_abc_file_to_1line(melody_path, is_melody=True)

            if not source:
                logger.info('%s is empty, skipped', str(chord_path))
                continue

            # melodyで休符が続く場合もskip
            # if 'z4 | z4' in reference:
            #     logger.info('%s has too long rest, skipped', str(melody_path))
            #     continue

            # 4/4以外は除外
            if 'M:4/4' not in source:
                logger.info('%s is not 4/4, skipped', str(melody_path))
                continue

            # データのprefixをつける。1 tokenになりそうな単位にする。
            if FLAGS.source_prefix:
                data_source_name = ''
                if 'weimar_jazz' in str(chord_path):
                    data_source_name = 'we'
                elif 'Wikifonia' in str(chord_path):
                    data_source_name += 'wiki'
                elif 'charlie' in str(chord_path):
                    data_source_name += 'char'
                elif 'fake' in str(chord_path):
                    data_source_name += 'fa'
                source = f'?{data_source_name}?' + source

            samples.append({
                'chord': source,
                'melody': reference,
                'source_path': str(chord_path),
                'source_music': chord_path.parent.parent.name
            })
    df = pd.DataFrame.from_records(samples)
    logger.info('there are %s samples', df.shape[0])
    df.to_json(output_filename,
               orient='records',
               force_ascii=False,
               lines=True)

    splitter = GroupShuffleSplit(test_size=FLAGS.valid_ratio,
                                 n_splits=1,
                                 random_state=0)
    split = splitter.split(df, groups=df['source_music'])
    train_indices, valid_indices = next(split)
    # train, valid = train_test_split(df,
    #                                 test_size=,
    #                                 random_state=0,
    #                                 groups=df['source_music'])
    train_df = df.loc[train_indices]
    valid_df = df.loc[valid_indices]
    logger.info('there are %s train samples', train_df.shape[0])

    train_df.to_json(output_filename.parent.joinpath('train_jsonl.json'),
                     orient='records',
                     force_ascii=False,
                     lines=True)
    logger.info('there are %s valid samples', valid_df.shape[0])
    valid_df.to_json(output_filename.parent.joinpath('valid_jsonl.json'),
                     orient='records',
                     force_ascii=False,
                     lines=True)

    # with output_filename.open('w') as fp:
    #     writer = jsonlines.Writer(fp)
    #     writer.write_all(samples)
    #     writer.close()


if __name__ == '__main__':
    app.run(main)
