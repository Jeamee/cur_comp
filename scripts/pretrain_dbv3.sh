BS=128
LR=1e-5
CORPUS=/workspace/corpus.txt
SAVE_DIR=/workspace/pretrain_dbv3
EPOCH=10
WP_STEPS=10000
MODEL="microsoft/deberta-v3-large"

mkdir $SAVE_DIR

python ../src/pretrain.py --model $MODEL \
--corpus $CORPUS \
--output $SAVE_DIR \
--bs $BS \
--epoch $EPOCH \
--lr $LR \
--warmup_steps $WP_STEPS