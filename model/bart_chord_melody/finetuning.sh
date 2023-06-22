PEAK_LR=3e-5
MAX_SENTENCES=2
SAVEPT="output/03_bart_chord_melody/generate"
UPDATE_FREQ=4
TENSORBOARD_DIR="output/03_bart_chord_melody/generate/tensorboard"
DATASET_DIR="data/v003_for_bart/generate"
MAX_UPDATE=80000
PRETRAINED_MODEL="output/03_bart_chord_melody/checkpoint_best.pt"

#CUDA_VISIBLE_DEVICES=0,1,2,3 fairseq-train $DATASET_DIR --arch bart_base --task denoising --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
#    --mask-length span-poisson --mask 0.35 --poisson-lambda 3.5 --permute-sentences -1 --replace-length 1 \
#    --tokens-per-sample 512 --max-sentences $MAX_SENTENCES  --rotate 0.0 \
#    --max-update $MAX_UPDATE --tensorboard-logdir $TENSORBOARD_DIR --update-freq $UPDATE_FREQ \
#    --optimizer adam --adam-betas '(0.9,0.98)' --adam-eps 1e-6 --clip-norm 0.0 \
#    --lr-scheduler polynomial_decay --lr $PEAK_LR --warmup-updates 10000 \
#    --dropout 0.1 --attention-dropout 0.1 --weight-decay 0.01 \
#    --seed 0 --log-format simple --log-interval 1 --save-dir $SAVEPT \
#    --num-workers 0 --fp16-init-scale 4 \
#    --layernorm-embedding --share-all-embeddings \
#    --skip-invalid-size-inputs-valid-test \
#    --encoder-embed-dim 480 \
#    --encoder-ffn-embed-dim 2048 \
#    --max-target-positions 1024 \
#    --max-source-positions 1024 \
#    --encoder-normalize-before --decoder-normalize-before \
#    --total-num-update $MAX_UPDATE --keep-best-checkpoints 2 --save-interval 1 --fp16

CUDA_VISIBLE_DEVICES=0,1,2,3 fairseq-train $DATASET_DIR --arch bart_base --restore-file $PRETRAINED_MODEL \
--save-dir $SAVEPT --tensorboard-logdir $TENSORBOARD_DIR \
--task translation --source-lang src --target-lang tgt \
--criterion label_smoothed_cross_entropy --dataset-impl raw \
--label-smoothing 0.1 \
--optimizer adam --adam-eps 1e-6 --adam-betas '{0.9, 0.98}' --lr-scheduler polynomial_decay --lr $PEAK_LR \
--warmup-updates 2500 --total-num-update $MAX_UPDATE --dropout 0.3 --attention-dropout 0.1  --weight-decay 0.0 \
--max-tokens 1024 --update-freq $UPDATE_FREQ --save-interval -1 --no-epoch-checkpoints --seed 0 --log-format simple --log-interval 2 \
--reset-optimizer --reset-meters --reset-dataloader --reset-lr-scheduler  --save-interval-updates 5000 \
--max-update $MAX_UPDATE --num-workers 0 \
--layernorm-embedding --share-all-embeddings \
--encoder-normalize-before --decoder-normalize-before \
--keep-best-checkpoints 2 --save-interval 2 --fp16 \
    --encoder-embed-dim 480 \
    --encoder-ffn-embed-dim 2048 \
    --max-target-positions 1024 \
    --max-source-positions 1024
