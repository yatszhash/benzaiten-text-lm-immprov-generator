"""
スクリプト
"""
from __future__ import print_function
# import gevent
# from gevent import monkey
# from gevent.threadpool import ThreadPool

# patches stdlib (including socket and ssl modules) to cooperate with other greenlets
# monkey.patch_all()

import pathlib

from absl import app
from absl import flags
import logging

from tqdm import tqdm

from preprocess.util import to_abc_notation

FLAGS = flags.FLAGS
flags.DEFINE_string('input', '', '入力')
flags.DEFINE_string('output', '', '出力')
flags.DEFINE_bool('use_cache', False, '')
flags.DEFINE_bool('original_xml2abc', False, '')

logger = logging.getLogger(__name__)


def from_all_segment_to_abc_notation(segment_root_dir,
                                     segment_abc_notation_root,
                                     use_cache=False,
                                     original_xml2abc=False):
    all_files = list(segment_root_dir.glob('**/*.xml')) + list(
        segment_root_dir.glob('**/*.mxl'))

    # jobs = [
    #     gevent.spawn(_f, segment_abc_notation_root, segment_root_dir)
    #     for path in all_files
    # ]

    # pool = ThreadPool(20)

    def _f(path):
        abc_filename = segment_abc_notation_root.joinpath(
            path.relative_to(segment_root_dir).with_suffix('.abc'))
        abc_filename.parent.mkdir(exist_ok=True, parents=True)
        if use_cache and abc_filename.exists():
            return
        to_abc_notation(path, abc_filename, original=original_xml2abc)

    #
    # for path in all_files:
    #     pool.spawn(_f, path)
    # gevent.wait()

    for path in tqdm(all_files):
        print(path)
        _f(path)


def main(unused_argv) -> None:
    """main関数"""
    from_all_segment_to_abc_notation(pathlib.Path(FLAGS.input),
                                     pathlib.Path(FLAGS.output),
                                     FLAGS.use_cache, FLAGS.original_xml2abc)


if __name__ == '__main__':
    app.run(main)
