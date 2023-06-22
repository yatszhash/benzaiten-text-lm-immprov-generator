PEAK_LR=3e-5
MAX_SENTENCES=2
SAVEPT="output/03_bart_chord_melody/generate/checkpoint_best.pt"
UPDATE_FREQ=4
TENSORBOARD_DIR="output/03_bart_chord_melody/generate/tensorboard"
DATASET_DIR="data/v003_for_bart/generate"
MAX_UPDATE=80000
#PRETRAINED_MODEL="output/03_bart_chord_melody/generate/checkpoint_best.pt"

fairseq-generate $DATASET_DIR --path $SAVEPT --task translation \
--dataset-impl raw --gen-subset valid --source-lang src --target-lang tgt --label-smoothing 0.1 \
--criterion label_smoothed_cross_entropy --max-tokens 1024