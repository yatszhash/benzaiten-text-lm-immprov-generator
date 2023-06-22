export TOKENIZERS_PARALLELISM=false
export HF_MLFLOW_LOG_ARTIFACTS=false
export ABC2MIDI=/usr/bin/abc2midi
export PYTHONPATH=.
OUTPUTPATH='output/late_submission'
INPUT_DIR='data/battle_files'
CUDA_VISIBLE_DEVICES=0 python bart/inference_and_post_process_text_to_music.py \
--output_dir $OUTPUTPATH/v005 \
--model_dir output/v005_02_text_to_music_ver2_filtered \
--backing_dir $INPUT_DIR \
--scale diatonic \
--replace_only_long \
--chord_duration_fine_grained \
--device cuda \
--original_xml2abc
CUDA_VISIBLE_DEVICES=0 python bart/inference_and_post_process_text_to_music.py \
--output_dir $OUTPUTPATH/v006 \
--model_dir output/v006_text_to_music_ver3_1_filtered \
--backing_dir $INPUT_DIR \
--scale diatonic \
--replace_only_long \
--device cuda \
--chord_duration_fine_grained
CUDA_VISIBLE_DEVICES=0 python bart/inference_and_post_process_text_to_music.py \
--output_dir $OUTPUTPATH/v006_not_finegrained \
--model_dir output/v006_text_to_music_ver3_1_filtered \
--backing_dir $INPUT_DIR \
--scale diatonic \
--replace_only_long \
--device cuda

