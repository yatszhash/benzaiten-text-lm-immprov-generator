export TOKENIZERS_PARALLELISM=false
export HF_MLFLOW_LOG_ARTIFACTS=false
CUDA_VISIBLE_DEVICES=0,1,2,3 deepspeed --num_gpus=4 model/bart_text_to_music/run_text_to_music_generation.py model/bart_text_to_music/train_config_v06.json