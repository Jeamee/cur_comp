set -Eeuox
DATE=0501
DECODER=softmax
BS=32
MAX_LEN=512
MODEL="microsoft/deberta-v2-xlarge"


for FOLD in 0 1 2 3 4;do
    DESC=pseudo-$FOLD
    python ../src/train.py --fold $FOLD --model $MODEL --decoder $DECODER --warmup_ratio 300 \
    --trans_lr 5e-6 --epochs 1 --max_len $MAX_LEN --batch_size $BS --valid_batch_size 32 \
    --clip_grad_norm 1.0 --gradient_ckpt --pseudo_csv /workspace/pseudo_oof${FOLD}.pkl --val_check_interval 300 \
    --desc $DESC --lr_decay 0.95 \
    --input_csv /workspace/train_folds.csv \
    --output /workspace/${MODEL#*/}_${DECODER}_bs${BS}_ml${MAX_LEN}_${DATE}_${DESC}_${FOLD}
done
