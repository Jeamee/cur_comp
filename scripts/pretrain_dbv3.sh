set -Eeuox
BS=16
LR=1e-5
CORPUS=/workspace/corpus.txt
SAVE_DIR=/workspace/pretrain_dbv3
EPOCH=100
MODEL="microsoft/deberta-v3-large"

mkdir $SAVE_DIR

python ../src/pretrain.py --model $MODEL \
--corpus $CORPUS \
--output $SAVE_DIR \
--bs $BS \
--epoch $EPOCH \
--lr $LR