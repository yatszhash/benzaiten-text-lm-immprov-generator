DATASET_DIR=data/v003_for_bart/generate
N_ALL=$(cat $DATASET_DIR/source.txt | wc -l)
echo $N_ALL
N_VALID=$((N_ALL * 2 / 10))
echo $N_VALID
N_TRAIN=$((N_ALL - N_VALID))
echo $N_TRAIN

head -n $N_TRAIN $DATASET_DIR/source.txt >| $DATASET_DIR/train.src-tgt.src
head -n $N_TRAIN $DATASET_DIR/reference.txt >| $DATASET_DIR/train.src-tgt.tgt
tail -n $N_VALID $DATASET_DIR/source.txt >| $DATASET_DIR/valid.src-tgt.src
tail -n $N_VALID $DATASET_DIR/reference.txt >| $DATASET_DIR/valid.src-tgt.tgt

#=train.src-tgt.src
#$DATASET_DIR/train.src-tgt.tgt
#cat $VALID_SRC | python3 ./jaBART_preprocess.py --bpe_model $SENTENCEPIECE_MODEL --bpe_dict $DICT > $DATASET_DIR/valid.src-tgt.src
#cat $VALID_TGT | python3 ./jaBART_preprocess.py --bpe_model $SENTENCEPIECE_MODEL --bpe_dict $DICT > $DATASET_DIR/valid.src-tgt.tgt