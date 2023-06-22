"""
予選用のmain script
"""
import itertools
import os
import pathlib
import re
import time

import music21
import torch
import transformers
from absl import app
from absl import flags
import logging

from bart.inference import from_all_chord_csvs_to_text2text_inputs, \
    merge_backing_midi_and_melody_abc
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from preprocess.util import extract_single_note_from_multiple_notes

MAX_LENGTH = 512
MIN_LENGTH = 60
music21.defaults.ticksPerQuarter = 480
# pycharmのバグでpathが渡らないことがある。
os.environ['PATH'] = f"{os.getenv('PATH')};/opt/homebrew/bin"

FLAGS = flags.FLAGS
flags.DEFINE_string('backing_dir', '', '入力')
flags.DEFINE_string('output_dir', '', '出力')
flags.DEFINE_string('hypothesis_filename', '', '')
flags.DEFINE_bool('only_original_pitch', False, 'オクターブ上げたものも出力するか？')
flags.DEFINE_string('model_dir', '', '')
flags.DEFINE_string('device', 'cuda', 'mpsでは動かない')

flags.DEFINE_bool('only_chord_change', False,
                  'chordが変更された時だけ追加する。train時と条件を揃えるにはTrueにしておく。')
flags.DEFINE_enum('source_prefix', '', ['', 'wiki', 'char', 'we'], '')
flags.DEFINE_bool('append_end_note', False, '終わりの音を追加するか？')
# flags.DEFINE_bool('input_key', False, 'keyを明治的にinputするか？')
flags.DEFINE_bool('chord_duration_fine_grained', False,
                  'chordのdurationを細かくするか？')
flags.DEFINE_enum('scale', '',
                  ['', 'blues', 'diatonic', 'melodic_minor', 'dorian'], '')
flags.DEFINE_bool('replace_only_long', False, '')
flags.DEFINE_enum('generation_method', 'beam',
                  ['beam', 'diverse_beam', 'beam_sampling'], '')
flags.DEFINE_bool('original_xml2abc', False,
                  'originalのxml2abcを呼ぶか？v005以前はこちらを指定しないと、うまく生成されない。')

flags.mark_flag_as_required('backing_dir')
flags.mark_flag_as_required('output_dir')
flags.register_multi_flags_validator(
    ['hypothesis_filename', 'model_dir'],
    multi_flags_checker=lambda value: (value['hypothesis_filename'] != '') !=
    (value['model_dir'] != ''),
    message=
    'please exactly select postprocess for hypothesis or in-place inference')

# SOURCE_PREFIXES = ['', 'we', 'wiki', 'char', 'fa']
# weは出力しない
# SOURCE_PREFIXES = ['we', 'wiki', 'char', 'fa'][1:]
SOURCE_PREFIXES = ['char', 'wiki']
SOURCE_PREFIXES_V006 = ['char', 'fa', 'wiki']
MODEL_NAME = 'v005'
# SOURCE_PREFIXES = [
#     'char',
# ]

logger = logging.getLogger(__name__)

# TODO beamとsamplingどっちが良い？ sampling安定しない。
beam_generate_config = {
    'num_beams': 4,
    'num_return_sequences': 4,
    # 'diverse_penalty': 2.0,
    # 'num_beam_groups': 4,
    # 'use_cache': True,
    'early_stopping': True,
    # 'temperature': 0.8
}

diverse_beam_generate_config = {
    'num_beams': 16,
    'num_return_sequences': 8,
    'diversity_penalty': 0.5,
    "num_beam_groups": 4,
    # "diversity_penalty": 1.0,
    'use_cache': True,
    'early_stopping': True
}

beam_sampling_config = {
    'num_beams': 4,
    'num_return_sequences': 4,
    'top_p': 0.75,
    'output_scores': True,
    # 'temperature': 2.0,
    # 'diverse_penalty': 2.0,
    # 'num_beam_groups': 4,
    # 'use_cache': True,
    'early_stopping': True,
    'do_sample': True
    # 'temperature': 0.8
}

sample_generate_config = {
    'top_p': 0.8,
    'temperature': 0.8,
    'do_sample': True,
    'use_cache': True,
    'early_stopping': True,
    'num_return_sequences': 4,
}

SAMPLING_CONFIG = {
    'beam': beam_generate_config,
    'diverse_beam': diverse_beam_generate_config
}

# spaceが入ると別token扱いにされるっぽい。
bad_words = ['z4', 'z2', ' z4', ' z2']

# NOTE force_wordsは時間がかかりすぎる'
# force_words = ['V:1\\n', 'L:1/4\\n', 'M:4/4\\n']


def model_load(model_dir: str, device: str):
    # TODO 高速化のためにあらかじめロードしておく
    model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)
    if device != 'cpu':
        model = model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    # generator = pipeline("text2text-generation",
    #                      model=model,
    #                      tokenizer=tokenizer,
    #                      device=device)

    return model, tokenizer


def generate(input_texts, model: transformers.PretrainedBartModel,
             tokenizer: transformers.PreTrainedTokenizer, generate_config):
    torch.manual_seed(0)
    # parallelだとうまく行かない可能性がある。
    input_tokens = tokenizer(input_texts,
                             return_tensors='pt',
                             padding=True,
                             max_length=MAX_LENGTH,
                             truncation=True)['input_ids'].to(model.device)
    # record = {
    #     f"{self.return_name}_text": self.tokenizer.decode(
    #         output_ids,
    #         skip_special_tokens=True,
    #         clean_up_tokenization_spaces=clean_up_tokenization_spaces,
    #     )
    # }
    # records.append(record)
    outputs = model.generate(
        input_tokens,
        return_dict_in_generate=True,
        max_length=MAX_LENGTH,
        min_length=MIN_LENGTH,
        bad_words_ids=tokenizer(bad_words, add_special_tokens=False).input_ids,
        # force_words_ids=tokenizer(force_words,
        #                           add_special_tokens=False).input_ids,
        **generate_config)
    print([len(seq) for seq in outputs['sequences']])
    outputs['candidates'] = tokenizer.batch_decode(
        outputs['sequences'],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True)

    num_return_sequences = sample_generate_config['num_return_sequences']
    if num_return_sequences > 1:
        candidates = []
        for i in range(len(outputs['candidates']) // num_return_sequences):
            candidates.append(
                outputs['candidates'][i * num_return_sequences:(i + 1) *
                                      num_return_sequences])
        outputs['candidates'] = candidates

    return outputs


def main(unused_argv) -> None:
    """main関数"""
    start_time = time.perf_counter()
    output_dir = pathlib.Path(FLAGS.output_dir)
    input_dir = pathlib.Path(FLAGS.backing_dir)
    backing_midi_paths = sorted(
        pathlib.Path(FLAGS.backing_dir).glob('**/*backing.mid'))
    # TODO コードを全種類チェックする。

    if 'v006' in FLAGS.model_dir:
        global SOURCE_PREFIXES
        SOURCE_PREFIXES = SOURCE_PREFIXES_V006
        global MODEL_NAME
        MODEL_NAME = 'v006'
    if not FLAGS.hypothesis_filename:
        model, tokenizer = model_load(FLAGS.model_dir, FLAGS.device)
        input_samples = from_all_chord_csvs_to_text2text_inputs(
            input_dir=input_dir,
            output_dir=output_dir.joinpath('input_samples'),
            only_chord_change=FLAGS.only_chord_change,
            fine_grained=FLAGS.chord_duration_fine_grained,
            original_xml2abc=FLAGS.original_xml2abc)
        # if FLAGS.input_key:
        #     keys = [
        #         re.search(r'\nK:(\w+)\n', line['chord']).group(1)
        #         for line in input_samples
        #     ]
        # else:
        #     keys = ['C'] * len(input_samples)
        keys = [sample['key'] for sample in input_samples]
        # TODO ここにsource_prefixのループを差し込む
        input_texts = []
        for sample in input_samples:
            inputs_for_current_sample = []
            for prefix in SOURCE_PREFIXES:
                source_prefix = f'?{prefix}?' if prefix else ''
                inputs_for_current_sample.append(source_prefix +
                                                 sample['chord'])
                # input_texts = [
                #     [source_prefix + sample['chord']]
                # ]
            input_texts.append(inputs_for_current_sample)

        sample_dim = len(input_texts)
        source_prefix_dim = len(input_texts[0])

        # 2次元だとgenerateがうけとらないので、一旦フラットにする
        input_texts = list(itertools.chain.from_iterable(input_texts))
        candidates_list = generate(
            input_texts=input_texts,
            model=model,
            tokenizer=tokenizer,
            generate_config=SAMPLING_CONFIG[FLAGS.generation_method])
        # json.dump(candidates_list,
        #           output_dir.joinpath('candidates.json').open('w'))
    # output_melody_midi_dir = output_dir.joinpath('melody_midi')
    # output_melody_midi_dir.mkdir(exist_ok=True, parents=True)
    #
    # output_melody_abc_dir = output_dir.joinpath('melody_abc')

    # hypotheses = bart_generate(bart, xml_paths, **eval_kwargs)
    else:
        # FIXME 複数candidatesある場合に合わせる。
        with open(FLAGS.hypothesis_file) as f:
            text = f.read()
            candidates_list = re.split(r'(?=L:1)', text)[1:]
        raise ValueError('not yet implemented for hypothesis file input')

    candidates_list = candidates_postprocess(candidates_list['candidates'])

    # 元の次元に戻す。
    candidates_list = [
        candidates_list[i * source_prefix_dim:(i + 1) * source_prefix_dim]
        for i in range(sample_dim)
    ]
    # abcs = [re.sub(r&) for abc in abcs]
    # abcs = [re.sub('K:\w+', 'K:C', abc) for abc in abcs]

    for i, (candidates_for_prefixes, backing_midi_path,
            key) in enumerate(zip(candidates_list, backing_midi_paths, keys)):
        assert len(candidates_for_prefixes) == len(SOURCE_PREFIXES)
        for candidates, prefix in zip(candidates_for_prefixes,
                                      SOURCE_PREFIXES):
            current_output_root = pathlib.Path(FLAGS.output_dir).joinpath(
                f'predicted/{backing_midi_path.stem}/{prefix}')
            current_output_root.mkdir(exist_ok=True, parents=True)
            for j, abc in enumerate(candidates):
                output_full_midi_path = current_output_root.joinpath(
                    f'{MODEL_NAME}_{prefix}_{j:02}_full.mid')
                output_melody_midi_path = current_output_root.joinpath(
                    f'{MODEL_NAME}_{prefix}_{j:02}_melody.mid')
                output_melody_abc_path = current_output_root.joinpath(
                    f'{MODEL_NAME}_{prefix}_{j:02}_melody.abc')
                # debugのため、開発中は一旦abcを保存しておく
                with output_melody_abc_path.open('w') as f:
                    f.write(abc)

                try:
                    merge_backing_midi_and_melody_abc(
                        abc,
                        backing_midi_path,
                        output_full_midi_path=output_full_midi_path,
                        output_melody_midi_path=output_melody_midi_path,
                        key=key,
                        scale=FLAGS.scale,
                        replace_only_long=FLAGS.replace_only_long)
                except Exception as e:
                    logger.error('failed to convert hypothesis %d for %s',
                                 j,
                                 backing_midi_path.stem,
                                 exc_info=e)
    end_time = time.perf_counter()
    print('process time: ', end_time - start_time)


def postprocess(candidate):
    # 1/4以外が生成されてしまうことがあるため
    # abcs = [re.sub('L:\d/\d', 'L:1/4', abc) for abc in abcs]
    # 1/4以外が生成されてしまうことがあるため
    # 重音が生成されることがあるため取り除く。
    candidate = extract_single_note_from_multiple_notes(candidate)
    candidate = 'X:1\n' + candidate
    candidate = re.sub('L:\d/\d', 'L:1/4', candidate)
    candidate = re.sub('M:\d/\d', 'M:4/4', candidate)
    candidate = re.sub('K:\w+', 'K:C', candidate)
    # 8小節以上生成されている場合、9小節以降を削除する。
    if candidate.count('|') >= 9:
        candidate = '|'.join(candidate.split('|')[:8]) + ' |'
    candidate = re.sub('K:C\n',
                       'K:C\n%%MIDI program 1 17\nz4 | z4 | z4 | z4 |',
                       candidate)
    candidate = re.sub(r'\|$', '| %12', candidate)
    return candidate


def candidates_postprocess(candidates_list):
    """
    candidatesは2次元が前提
    :param candidates_list:
    :return:
    """
    return [[postprocess(candidate) for candidate in candidates]
            for candidates in candidates_list]
    #
    # # 1/4以外が生成されてしまうことがあるため
    # # abcs = [re.sub('L:\d/\d', 'L:1/4', abc) for abc in abcs]
    # # 1/4以外が生成されてしまうことがあるため
    # candidates = ['X:1\n' + abc for abc in candidates]
    # candidates = [re.sub('L:\d/\d', 'L:1/4', abc) for abc in candidates]
    # candidates = [re.sub('K:\w+', 'K:C', abc) for abc in candidates]
    # candidates = [
    #     re.sub('K:C\n', 'K:C\n%%MIDI program 1 17\nz4 | z4 | z4 | z4 |', abc)
    #     for abc in candidates
    # ]
    # candidates = [re.sub(r'\|$', '| %12', abc) for abc in candidates]
    # return candidates


if __name__ == '__main__':
    app.run(main)
