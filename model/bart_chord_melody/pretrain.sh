PEAK_LR=1e-4
MAX_SENTENCES=2
SAVEPT="output/03_bart_chord_melody"
UPDATE_FREQ=4
TENSORBOARD_DIR="output/03_bart_chord_melody/tensorboard"
DATASET_DIR="data/v003_for_bart"
MAX_UPDATE=500000

CUDA_VISIBLE_DEVICES=0,1,2,3 fairseq-train $DATASET_DIR --arch bart_base --task denoising --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --mask-length span-poisson --mask 0.35 --poisson-lambda 3.5 --permute-sentences -1 --replace-length 1 \
    --tokens-per-sample 512 --max-sentences $MAX_SENTENCES  --rotate 0.0 \
    --max-update $MAX_UPDATE --tensorboard-logdir $TENSORBOARD_DIR --update-freq $UPDATE_FREQ \
    --optimizer adam --adam-betas '(0.9,0.98)' --adam-eps 1e-6 --clip-norm 0.0 \
    --lr-scheduler polynomial_decay --lr $PEAK_LR --warmup-updates 10000 \
    --dropout 0.1 --attention-dropout 0.1 --weight-decay 0.01 \
    --seed 0 --log-format simple --log-interval 1 --save-dir $SAVEPT \
    --num-workers 0 --fp16-init-scale 4 \
    --layernorm-embedding --share-all-embeddings \
    --skip-invalid-size-inputs-valid-test \
    --encoder-embed-dim 480 \
    --encoder-ffn-embed-dim 2048 \
    --max-target-positions 1024 \
    --max-source-positions 1024 \
    --encoder-normalize-before --decoder-normalize-before \
    --total-num-update $MAX_UPDATE --keep-best-checkpoints 2 --save-interval 1 --fp16 \
    --no-epoch-checkpoints




#TOTAL_NUM_UPDATES=20000
#WARMUP_UPDATES=500
#LR=3e-05
#MAX_TOKENS=2048
#UPDATE_FREQ=4
#BART_PATH=/path/to/bart/model.pt
#
#CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 fairseq-train  \
#    --restore-file $BART_PATH \
#    --max-tokens $MAX_TOKENS \
#    --task denoising \
#    --source-lang source --target-lang target \
#    --truncate-source \
#    --layernorm-embedding \
#    --share-all-embeddings \
#    --share-decoder-input-output-embed \
#    --reset-optimizer --reset-dataloader --reset-meters \
#    --required-batch-size-multiple 1 \
#    --arch mbart_base \
#    --criterion label_smoothed_cross_entropy \
#    --label-smoothing 0.1 \
#    --dropout 0.1 --attention-dropout 0.1 \
#    --weight-decay 0.01 --optimizer adam --adam-betas "(0.9, 0.999)" --adam-eps 1e-08 \
#    --clip-norm 0.1 \
#    --lr-scheduler polynomial_decay --lr $LR --total-num-update $TOTAL_NUM_UPDATES --warmup-updates $WARMUP_UPDATES \
#    --fp16 --update-freq $UPDATE_FREQ \
#    --skip-invalid-size-inputs-valid-test \
#    --find-unused-parameters;


