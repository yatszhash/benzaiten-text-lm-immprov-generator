import multiprocessing
import pathlib
import traceback

from absl import app
from absl import flags
import logging

from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

from preprocess.util import to_abc_notation, transpose_and_save_xml_score

FLAGS = flags.FLAGS
flags.DEFINE_string('input_dir', '', '入力')
flags.DEFINE_string('output_dir', '', '出力')
flags.DEFINE_bool('write_chord_notes', False, 'chordの構成音を書き出すか')
flags.DEFINE_bool('only_four_beat', False, '')
# flags.DEFINE_enum('format', 'xml', ['xml', 'midi'], 'format')

logger = logging.getLogger(__name__)


def main(unused_argv) -> None:
    """main関数"""
    input_dir = pathlib.Path(FLAGS.input_dir)
    output_dir = pathlib.Path(FLAGS.output_dir)

    # if FLAGS.format == 'midi':
    #     extension = FLAGS.format[:-1]
    # else:
    #     extension = FLAGS.format
    all_input_paths = list(input_dir.glob(f'**/*.xml')) + list(
        input_dir.glob(f'**/*.mid')) + list(input_dir.glob(f'**/*.mxl'))
    all_output_paths = [
        output_dir.joinpath(path.relative_to(input_dir))
        for path in all_input_paths
    ]

    logger.info('total %d files', len(all_input_paths))
    process_map(_process_path,
                list(
                    zip(all_input_paths, all_output_paths,
                        [FLAGS.write_chord_notes] * len(all_input_paths),
                        [FLAGS.only_four_beat] * len(all_input_paths))),
                max_workers=10)
    # for paths in zip(all_input_paths, all_output_paths,
    #                  [FLAGS.write_chord_notes] * len(all_input_paths)):
    #     _process_path(paths)
    # with multiprocessing.Pool(20) as pool:
    #     imap = pool.apply_async(_process_path,
    #                             list(zip(all_input_paths, all_output_paths)))
    # result = list(tqdm(imap, total=len(all_input_paths)))
    # for path in tqdm(all_input_paths):
    #     _process_path(input_dir, output_dir, path)

    # for path in tqdm(input_dir.glob(f'**/*.mid')):
    #     logger.info('processing %s', str(path))
    #     output_path = output_dir.joinpath(path.relative_to(input_dir))
    #     output_path.parent.mkdir(exist_ok=True, parents=True)
    #     transpose_and_save_xml_score(path,
    #                                  output_path,
    #                                  save_format='midi',
    #                                  only_four_beat=False)


def _process_path(paths):
    path, output_path, write_chord_notes, only_four_beat = paths
    logger.info('processing %s', str(path))
    print('processing %s', str(path))
    output_path.parent.mkdir(exist_ok=True, parents=True)
    try:
        if output_path.exists():
            print(f'{str(output_path)} exists, skipped')
            return
        if path.suffix == '.mid':
            transpose_and_save_xml_score(path,
                                         output_path,
                                         save_format='midi',
                                         only_four_beat=only_four_beat,
                                         write_chord_notes=write_chord_notes)
        elif path.suffix == '.mxl':
            transpose_and_save_xml_score(path,
                                         output_path.with_suffix('.xml'),
                                         save_format='xml',
                                         only_four_beat=only_four_beat,
                                         write_chord_notes=write_chord_notes)
        else:
            transpose_and_save_xml_score(path,
                                         output_path,
                                         save_format=path.suffix[1:],
                                         only_four_beat=only_four_beat,
                                         write_chord_notes=write_chord_notes)
    except Exception as e:
        print(str(path), e.__repr__(), traceback.format_exc())


if __name__ == '__main__':
    app.run(main)
