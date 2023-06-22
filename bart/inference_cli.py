"""
スクリプト
"""
import logging
import pathlib

from absl import app
from absl import flags

from bart.inference import generate_melodies
from gpt3.inference import from_chord_csv_to_chord_score

FLAGS = flags.FLAGS
flags.DEFINE_string('model_dir', '', '入力')
# flags.DEFINE_string('model_name', None, 'モデル名')
flags.DEFINE_string('output_dir', '', '出力')
# flags.DEFINE_string('backing_midi_dir', '', '')
flags.DEFINE_string('input_dir', '', '')
# flags.DEFINE_integer('n_candidates', 5, '')

logger = logging.getLogger(__name__)


def main(unused_argv) -> None:
    """main関数"""
    input_dir = pathlib.Path(FLAGS.input_dir)
    output_dir = pathlib.Path(FLAGS.output_dir)
    # backing_midi_dir = pathlib.Path(FLAGS.backing_midi_dir)

    generate_kwargs = {
        "beam": 4,
        "temperature": 1.0,
    }

    xml_dir = output_dir.joinpath('chord_score')
    for path in input_dir.glob('**/*.csv'):
        score = from_chord_csv_to_chord_score(path,
                                              rest=False,
                                              chord_notes=True)
        score_path = xml_dir.joinpath(str(
            path.relative_to(input_dir))).with_suffix('.xml')

        score_path.parent.mkdir(exist_ok=True, parents=True)
        score.write('xml', score_path)

    generate_melodies(FLAGS.model_dir, 'checkpoint_best.pt', xml_dir,
                      input_dir, output_dir, generate_kwargs)
    # prompts = []
    #
    # backing_paths = sorted(input_dir.glob('**/*_backing.mid'))
    # chord_paths = sorted(input_dir.glob('**/*_chord.csv.txt'))
    # for path in tqdm(chord_paths):
    #     score = from_chord_csv_to_chord_score(path)
    #     score_path = intermediate_dir.joinpath(
    #         'score/' + str(path.relative_to(input_dir))).with_suffix('.xml')
    #     score_path.parent.mkdir(exist_ok=True, parents=True)
    #     score.write('xml', score_path)
    #     abc_path = intermediate_dir.joinpath(
    #         'abc/' + str(path.relative_to(input_dir))).with_suffix('.abc')
    #     abc_path.parent.mkdir(exist_ok=True, parents=True)
    #     to_abc_notation(score_path, abc_path)
    #     prompts.append(generate_inference_prompt_from_file(abc_path))
    #
    # responses = []
    # for prompt, chord_path, backing_path in zip(prompts, chord_paths,
    #                                             backing_paths):
    #     logger.info('generating melody for %s', str(chord_path))
    #     responses.append([])
    #     for i in range(FLAGS.n_candidates):
    #         print("\n  Generating Song", i)
    #         response = openai.Completion.create(model=FLAGS.model_name,
    #                                             prompt=prompt,
    #                                             stop=" $ <end>",
    #                                             temperature=0.75,
    #                                             top_p=1.0,
    #                                             frequency_penalty=0.0,
    #                                             presence_penalty=0.0,
    #                                             max_tokens=1000)
    #         abc_notation = from_gpt3_response_to_abc_notation(
    #             response, f'{chord_path.stem}_{i:03}', FLAGS.model_name)
    #         score = music21.converter.parseData(abc_notation, format='abc')
    #         # TODO たまに不正なMIDIが生成される。
    #         current_score_path = output_dir.joinpath(
    #             chord_path.stem).joinpath(f'{i:03}_melody').with_suffix('.xml')
    #         current_score_path.parent.mkdir(exist_ok=True, parents=True)
    #         score.write('xml', current_score_path)
    #
    #         current_midi_path = current_score_path.with_suffix('.mid')
    #         score.write('midi', current_midi_path)
    #
    #         current_full_midi_path = current_midi_path.parent.joinpath(
    #             current_midi_path.stem.replace('melody',
    #                                            'full')).with_suffix('.mid')
    #
    #         merge_backing_midi_and_melody_midi(
    #             input_backing_midi_path=backing_path,
    #             output_melody_midi_path=current_midi_path,
    #             output_full_midi_path=current_full_midi_path)


if __name__ == '__main__':
    app.run(main)
