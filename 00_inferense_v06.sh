export TOKENIZERS_PARALLELISM=false
export HF_MLFLOW_LOG_ARTIFACTS=false
export ABC2MIDI=/usr/bin/abc2midi
export PYTHONPATH=.
CUDA_VISIBLE_DEVICES=0 python bart/inference_and_post_process_text_to_music.py \
--output_dir output/v006_text_to_music_ver3_1_filtered/test/beam_4_diatonic_scale_not_finegrained \
--model_dir output/v006_text_to_music_ver3_1_filtered \
--backing_dir input_dir \
--scale diatonic \
--replace_only_long \
--chord_duration_fine_grained \
--device cuda