import gc
gc.enable()

import sys
import ast
import argparse
import os
import random
import warnings
import logging
import time
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import pandas as pd
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
import torch
import torch.nn as nn
import bitsandbytes as bnb
import pytorch_lightning as pl

from tqdm import tqdm
from math import ceil
from copy import deepcopy
from sklearn import metrics
from torch.nn.parameter import Parameter
from torch.utils.data import DataLoader
from torch.nn import functional as F
from torch.optim.lr_scheduler import StepLR, LinearLR
from transformers import AdamW, AutoConfig, AutoModel, AutoTokenizer, get_cosine_schedule_with_warmup
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger


from utils import GradualWarmupScheduler, ReduceLROnPlateau, span_decode, Freeze
from model.model import NBMEModel
from data.dataset import TrainDataset
from loss.dice_loss import DiceLoss
from loss.focal_loss import FocalLoss
from loss.sce import SCELoss


warnings.filterwarnings("ignore")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--seed", type=int, default=43, required=False)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--desc", type=str, default="base", required=False)
    parser.add_argument("--trans_lr", type=float, required=True)
    parser.add_argument("--dynamic_merge_layers", action="store_true", required=False)
    parser.add_argument("--merge_layers_num", type=int, default=-2, required=False)
    parser.add_argument("--finetune", action="store_true", required=False)
    parser.add_argument("--output", type=str, default="../model", required=False)
    parser.add_argument("--input_csv", type=str, default="", required=True)
    parser.add_argument("--ckpt", type=str, default="", required=False)
    parser.add_argument("--max_len", type=int, default=510, required=False)
    parser.add_argument("--batch_size", type=int, default=8, required=False)
    parser.add_argument("--valid_batch_size", type=int, default=8, required=False)
    parser.add_argument("--epochs", type=int, default=20, required=False)
    parser.add_argument("--accumulation_steps", type=int, default=1, required=False)
    parser.add_argument("--loss", type=str, default="ce", required=False)
    parser.add_argument("--label_smooth", type=float, default=0.0, required=False)
    parser.add_argument("--warmup_ratio", type=float, default=0.05, required=False)
    parser.add_argument("--sce_alpha", type=float, required=False)
    parser.add_argument("--sce_beta", type=float, required=False)
    parser.add_argument("--decoder", type=str, default="softmax", required=False)
    parser.add_argument("--freeze", type=int, default=10, required=False)
    parser.add_argument("--freeze_method", type=str, default="hard", required=False)
    parser.add_argument("--gradient_ckpt", action="store_true", required=False)
    parser.add_argument("--clip_grad_norm", type=float, default=1.0, required=False)
    parser.add_argument("--lr_decay", type=float, default=1.0, required=False)
    parser.add_argument("--add_return_token", action="store_true", required=False)

    
    return parser.parse_args()


if __name__ == "__main__":
    NUM_JOBS = os.cpu_count() - 1
    args = parse_args()
    pl.seed_everything(seed=args.seed, workers=True)
    os.makedirs(args.output, exist_ok=True)
    df = pd.read_csv(args.input_csv)
    df['annotation'] = df['annotation'].apply(ast.literal_eval)
    df['location'] = df['location'].apply(ast.literal_eval)

    train_df = df[df["kfold"] != args.fold].reset_index(drop=True)
    valid_df = df[df["kfold"] == args.fold].reset_index(drop=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
        
    if args.add_return_token:
        tokenizer.add_tokens("\n", special_tokens=True)
        tokenizer.save_pretrained(args.output)

    for text_col in ['pn_history']:
        pn_history_lengths = []
        tk0 = tqdm(df[text_col].fillna("").values, total=len(df))
        for text in tk0:
            length = len(tokenizer(text, add_special_tokens=False)['input_ids'])
            pn_history_lengths.append(length)

    for text_col in ['feature_text']:
        features_lengths = []
        tk0 = tqdm(df[text_col].fillna("").values, total=len(df))
        for text in tk0:
            length = len(tokenizer(text, add_special_tokens=False)['input_ids'])
            features_lengths.append(length)

    args.max_len = max(pn_history_lengths) + max(features_lengths) + 3 # cls & sep & sep
   
    train_dataset = DataLoader(
            TrainDataset(tokenizer, args.max_len, train_df),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=3,
            pin_memory=True,
            )
    valid_dataset = DataLoader(
            TrainDataset(tokenizer, args.max_len, valid_df),
            batch_size=args.valid_batch_size,
            shuffle=False,
            num_workers=3,
            pin_memory=True,
            )

    num_train_steps = int(len(train_dataset) / args.batch_size / args.accumulation_steps * args.epochs)
    
    num_labels = 1
    span_num_labels = 2
    model = NBMEModel(
        max_len=args.max_len,
        model_name=args.model,
        num_train_steps=num_train_steps,
        transformer_learning_rate=args.trans_lr,
        dynamic_merge_layers=args.dynamic_merge_layers,
        merge_layers_num=args.merge_layers_num,
        num_labels=num_labels,
        span_num_labels=span_num_labels,
        steps_per_epoch=len(train_dataset) / args.batch_size,
        loss=args.loss,
        sce_alpha=args.sce_alpha,
        sce_beta=args.sce_beta,
        label_smooth=args.label_smooth,
        decoder=args.decoder,
        warmup_ratio=args.warmup_ratio,
        finetune=args.finetune,
        gradient_ckpt=args.gradient_ckpt,
        lr_decay=args.lr_decay,
    )
    

    if args.add_return_token:
        model.transformer.resize_token_embeddings(len(tokenizer))
    
    if args.ckpt:
        model.load(args.ckpt, weights_only=True, strict=False)

    early_stop_callback = EarlyStopping(monitor="valid/f1", min_delta=0.00, patience=12, verbose=True, mode="max")
    model_ckpt_callback = ModelCheckpoint(
            dirpath=args.output,
            monitor="valid/f1",
            mode="max",
            save_weights_only=True,
            filename="{epoch}-{valid/f1:.3f}",
            )
    model_ckpt_callback.FILE_EXTENSION = f".oof{args.fold}.bin"
    freeze = Freeze(epochs=args.freeze, method=args.freeze_method)

    logger = WandbLogger(name=f"{args.model}-fold{args.fold}-{args.desc}",
            project="NBME",
            )

    logger.experiment.config.update(vars(args))

    trainer = pl.Trainer(
            benchmark=True,
            logger=logger,
            deterministic=False,
            accelerator="gpu",
            gpus=1,
            auto_select_gpus=True,
            precision=16,
            gradient_clip_val=args.clip_grad_norm,
            log_gpu_memory=True,
            log_every_n_steps=5,
            enable_progress_bar=True,
            max_epochs=args.epochs,
            val_check_interval=0.25,
            callbacks=[freeze, model_ckpt_callback, early_stop_callback]
            )

    trainer.fit(model=model, train_dataloaders=train_dataset, val_dataloaders=valid_dataset)

        
    
