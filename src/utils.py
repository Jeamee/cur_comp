import copy
import os
import logging
import sys
sys.path.append("/workspace/tez")

import numpy as np
import pandas as pd
import torch
from math import ceil
from joblib import Parallel, delayed
from tez import enums
from tez.callbacks import Callback
from tqdm import tqdm

from torch.optim.lr_scheduler import _LRScheduler, ReduceLROnPlateau
from torch.optim import Optimizer
from torch._six import inf
from torch.nn.utils.rnn import pad_sequence


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


target_id_map = {
    "B-Lead": 0,
    "I-Lead": 1,
    "B-Position": 2,
    "I-Position": 3,
    "B-Evidence": 4,
    "I-Evidence": 5,
    "B-Claim": 6,
    "I-Claim": 7,
    "B-Concluding Statement": 8,
    "I-Concluding Statement": 9,
    "B-Counterclaim": 10,
    "I-Counterclaim": 11,
    "B-Rebuttal": 12,
    "I-Rebuttal": 13,
    "O": 14,
    "PAD": -100,
}
id_target_map = {v: k for k, v in target_id_map.items()}


span_target_id_map = {
                "Lead": 1,
                "Position": 2,
                "Evidence": 3,
                "Claim": 4,
                "Concluding Statement": 5,
                "Counterclaim": 6,
                "Rebuttal": 7
            }
span_id_target_map = {v: k for k, v in span_target_id_map.items()}





def biaffine_decode(logits):
    logits = torch.softmax(logits, -1)
    probs = torch.max(logits, -1).values.cpu().tolist()
    preds = torch.argmax(logits, -1).cpu().tolist()
    
    results = []
    for i, (pred, prob) in enumerate(zip(preds, probs)):
        tmp_preds = [[0] * 15 for _ in range(len(pred))]
        for row_idx, row in enumerate(pred):
            col_idx = 0
            while col_idx < len(pred):
                val = row[col_idx]
                if val == 0:
                    col_idx += 1
                    continue
                
                next_idx = col_idx + 1
                while next_idx < len(pred):
                    next_val = row[next_idx]
                    if val == next_val:
                        next_idx += 1
                        continue
                    break
                
                next_idx -= 1
                start, end = row_idx, next_idx
                label_str = span_id_target_map[val]
                cur_prob = prob[row_idx][next_idx]
                tmp_preds[start][target_id_map[f"B-{label_str}"]] = cur_prob
                if start == end:
                    col_idx = next_idx + 1
                    continue
                for idx in range(start + 1, end + 1):
                    tmp_preds[idx][target_id_map[f"I-{label_str}"]] = cur_prob
                
                col_idx = next_idx + 1
            
        results.append(tmp_preds)
        
    results = torch.tensor(results)
    return results
                    

def span_decode(start_logits, end_logits):
    start_logits = torch.softmax(start_logits, -1)
    end_logits = torch.softmax(end_logits, -1)
    
    start_probs = torch.max(start_logits, -1)
    end_probs = torch.max(end_logits, -1)
    
    start_preds = torch.argmax(start_logits, -1)
    end_preds = torch.argmax(end_logits, -1)
    
    start_preds = start_preds.cpu().tolist()
    end_preds = end_preds.cpu().tolist()
    start_probs = start_probs.values.cpu().tolist()
    end_probs = end_probs.values.cpu().tolist()
    
    preds = []
    for start_pred, end_pred, start_prob, end_prob in zip(start_preds, end_preds, start_probs, end_probs):
        pred = [[0] * 15 for _ in range(len(start_pred))]
        #pred = [14] * len(start_pred)
        idx = 0
        next_end_idx = None
        end_idx = None
        while idx < len(start_pred):
            s_type = start_pred[idx]
            if s_type == 0 and next_end_idx is None:
                pred[idx][14] = 1
                idx += 1
                continue
            elif s_type == 0 and next_end_idx is not None:
                idx += 1
                if next_end_idx == idx:
                    next_end_idx = None
                continue
            
            cur_type_str = span_id_target_map[s_type]
            cur_start_prob = start_prob[idx]
            cur_end_prob = 0
            
            
            end_idx = idx + 1
            while end_idx < len(end_pred):
                if end_pred[end_idx] == 0:
                    end_idx += 1
                    continue
                    
                e_type = end_pred[end_idx]
                end_idx += 1
                if s_type == e_type:
                    cur_end_prob = end_prob[end_idx - 1]
                    break

            prob = (cur_start_prob + cur_end_prob) / 2

            pred[idx][target_id_map[f"B-{cur_type_str}"]] = prob
            for i in range(idx + 1, end_idx):
                pred[i][target_id_map[f"I-{cur_type_str}"]] = prob
            idx += 1
            next_end_idx = end_idx
        preds.append(pred)
        
    preds = torch.tensor(preds)
    return preds


def smooth_one_hot(true_labels: torch.Tensor, classes: int, smoothing=0.0):
    """
    if smoothing == 0, it's one-hot method
    if 0 < smoothing < 1, it's smooth method

    """
    assert 0 <= smoothing < 1
    confidence = 1.0 - smoothing
    label_shape = torch.Size((true_labels.size(0), classes))
    with torch.no_grad():
        true_dist = torch.empty(size=label_shape, device=true_labels.device)
        true_dist.fill_(smoothing / (classes - 1))
        true_dist.scatter_(1, true_labels.data.unsqueeze(1), confidence)
    return true_dist


def _prepare_training_data_helper(args, tokenizer, df, train_ids, id_text_map):
    training_samples = []
    for idx in tqdm(train_ids):
        text = id_text_map[idx]
        encoded_text = tokenizer.encode_plus(
            text,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
        input_ids = encoded_text["input_ids"]
        input_labels = copy.deepcopy(input_ids)
        offset_mapping = encoded_text["offset_mapping"]

        for k in range(len(input_labels)):
            input_labels[k] = "O"

        sample = {
            "id": idx,
            "input_ids": input_ids,
            "text": text,
            "offset_mapping": offset_mapping,
        }

        temp_df = df[df["id"] == idx]
        for _, row in temp_df.iterrows():
            text_labels = [0] * len(text)
            discourse_start = int(row["new_start"])
            discourse_end = int(row["new_end"])
            prediction_label = row["discourse_type"]
            text_labels[discourse_start:discourse_end] = [1] * (discourse_end - discourse_start)
            target_idx = []
            for map_idx, (offset1, offset2) in enumerate(encoded_text["offset_mapping"]):
                if sum(text_labels[offset1:offset2]) > 0:
                    if len(text[offset1:offset2].split()) > 0:
                        target_idx.append(map_idx)

            targets_start = target_idx[0]
            targets_end = target_idx[-1]
            pred_start = "B-" + prediction_label
            pred_end = "I-" + prediction_label
            input_labels[targets_start] = pred_start
            input_labels[targets_start + 1 : targets_end + 1] = [pred_end] * (targets_end - targets_start)

        sample["input_ids"] = input_ids
        sample["input_labels"] = input_labels
        training_samples.append(sample)
    return training_samples


def read_txts(args, ids):
    id_text_map = {}
    for idx in tqdm(ids):
        filename = os.path.join(args.input, "train", idx + ".txt")
        with open(filename, "r") as f:
            text = f.read()
        
        text = text.replace("\xa0", " ")
        id_text_map[idx] = text
    
    return id_text_map


def prepare_training_data(df, tokenizer, args, num_jobs, only_bigger_than_1024=False):
    training_samples = []
    train_ids = df["id"].unique().tolist()
    
    id_text_map = {}
    
    train_ids_splits = np.array_split(train_ids, num_jobs)
    results = Parallel(n_jobs=num_jobs, backend="multiprocessing")(
        delayed(read_txts)(args, idx) for idx in train_ids_splits
    )
    for result in results:
        id_text_map.update(result)
    
    if only_bigger_than_1024:
        logging.info("only train on length > 1024")
        train_ids = filter(lambda x: len(id_text_map[x]) > 1024, train_ids)
    train_ids = sorted(train_ids, key=lambda x: len(id_text_map[x]))
    train_ids_splits = np.array_split(train_ids, num_jobs)
    results = Parallel(n_jobs=num_jobs, backend="multiprocessing")(
        delayed(_prepare_training_data_helper)(args, tokenizer, df, idx, id_text_map) for idx in train_ids_splits
    )
    for result in results:
        training_samples.extend(result)

    return training_samples


def calc_overlap(row):
    """
    Calculates the overlap between prediction and
    ground truth and overlap percentages used for determining
    true positives.
    """
    set_pred = set(row.predictionstring_pred.split(" "))
    set_gt = set(row.predictionstring_gt.split(" "))
    # Length of each and intersection
    len_gt = len(set_gt)
    len_pred = len(set_pred)
    inter = len(set_gt.intersection(set_pred))
    overlap_1 = inter / len_gt
    overlap_2 = inter / len_pred
    return [overlap_1, overlap_2]


def score_feedback_comp_micro(pred_df, gt_df):
    """
    A function that scores for the kaggle
        Student Writing Competition

    Uses the steps in the evaluation page here:
        https://www.kaggle.com/c/feedback-prize-2021/overview/evaluation
    This code is from Rob Mulla's Kaggle kernel.
    """
    gt_df = gt_df[["id", "discourse_type", "predictionstring"]].reset_index(drop=True).copy()
    pred_df = pred_df[["id", "class", "predictionstring"]].reset_index(drop=True).copy()
    pred_df["pred_id"] = pred_df.index
    gt_df["gt_id"] = gt_df.index
    # Step 1. all ground truths and predictions for a given class are compared.
    joined = pred_df.merge(
        gt_df,
        left_on=["id", "class"],
        right_on=["id", "discourse_type"],
        how="outer",
        suffixes=("_pred", "_gt"),
    )
    joined["predictionstring_gt"] = joined["predictionstring_gt"].fillna(" ")
    joined["predictionstring_pred"] = joined["predictionstring_pred"].fillna(" ")

    joined["overlaps"] = joined.apply(calc_overlap, axis=1)

    # 2. If the overlap between the ground truth and prediction is >= 0.5,
    # and the overlap between the prediction and the ground truth >= 0.5,
    # the prediction is a match and considered a true positive.
    # If multiple matches exist, the match with the highest pair of overlaps is taken.
    joined["overlap1"] = joined["overlaps"].apply(lambda x: eval(str(x))[0])
    joined["overlap2"] = joined["overlaps"].apply(lambda x: eval(str(x))[1])

    joined["potential_TP"] = (joined["overlap1"] >= 0.5) & (joined["overlap2"] >= 0.5)
    joined["max_overlap"] = joined[["overlap1", "overlap2"]].max(axis=1)
    tp_pred_ids = (
        joined.query("potential_TP")
        .sort_values("max_overlap", ascending=False)
        .groupby(["id", "predictionstring_gt"])
        .first()["pred_id"]
        .values
    )

    # 3. Any unmatched ground truths are false negatives
    # and any unmatched predictions are false positives.
    fp_pred_ids = [p for p in joined["pred_id"].unique() if p not in tp_pred_ids]

    matched_gt_ids = joined.query("potential_TP")["gt_id"].unique()
    unmatched_gt_ids = [c for c in joined["gt_id"].unique() if c not in matched_gt_ids]

    # Get numbers of each type
    TP = len(tp_pred_ids)
    FP = len(fp_pred_ids)
    FN = len(unmatched_gt_ids)
    # calc microf1
    my_f1_score = TP / (TP + 0.5 * (FP + FN))
    return my_f1_score


def score_feedback_comp(pred_df, gt_df, return_class_scores=False):
    class_scores = {}
    pred_df = pred_df[["id", "class", "predictionstring"]].reset_index(drop=True).copy()
    for discourse_type, gt_subset in gt_df.groupby("discourse_type"):
        pred_subset = pred_df.loc[pred_df["class"] == discourse_type].reset_index(drop=True).copy()
        class_score = score_feedback_comp_micro(pred_subset, gt_subset)
        class_scores[discourse_type] = class_score
    f1 = np.mean([v for v in class_scores.values()])
    if return_class_scores:
        return f1, class_scores
    return f1


class FeedbackDatasetValid:
    def __init__(self, samples, max_len, tokenizer, bigbird_padding=False):
        self.samples = samples
        self.max_len = max_len
        self.tokenizer = tokenizer
        self.length = len(samples)
        self.bigbird_padding = bigbird_padding
        
        self._init()
    
    def _init(self):
        new_samples = []
        for sample in self.samples:
            input_ids = sample["input_ids"]
            input_ids = [self.tokenizer.cls_token_id] + input_ids
            attention_mask = [1] * len(input_ids)

            if len(input_ids) > self.max_len - 1:
                input_ids = input_ids[: self.max_len - 1]
                attention_mask = attention_mask[: self.max_len - 1]
            
            input_ids = input_ids + [self.tokenizer.sep_token_id]
            attention_mask = attention_mask + [1]
            
            if self.bigbird_padding:
                if len(input_ids) > 1024:
                    padding_length = ceil(len(input_ids) / 64) * 64 - len(input_ids)
                else:
                    padding_length = 1024 - len(input_ids)
                input_ids = input_ids + [self.tokenizer.pad_token_id] * padding_length
                attention_mask = attention_mask + [0] * padding_length

            sample = {
                "ids": input_ids,
                "mask": attention_mask,
            }
            
            new_samples.append(sample)
        
        self.samples = new_samples
        
    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return self.samples[idx]

class Collate:
    def __init__(self, tokenizer, ):
        self.tokenizer = tokenizer

    def __call__(self, batch):
        output = dict()
        output["ids"] = pad_sequence([torch.tensor(sample["ids"], dtype=torch.long) for sample in batch], batch_first=True)
        output["mask"] = pad_sequence([torch.tensor(sample["mask"], dtype=torch.long) for sample in batch], batch_first=True)

        # calculate max token length of this batch
        #batch_max = max([len(ids) for ids in output["ids"]])

        # add padding
        #if self.tokenizer.padding_side == "right":
        #output["ids"] = [s + (batch_max - len(s)) * [self.tokenizer.pad_token_id] for s in output["ids"]]
        #output["mask"] = [s + (batch_max - len(s)) * [0] for s in output["mask"]]
        #else:
        #    output["ids"] = [(batch_max - len(s)) * [self.tokenizer.pad_token_id] + s for s in output["ids"]]
        #   output["mask"] = [(batch_max - len(s)) * [0] + s for s in output["mask"]]

        # convert to tensors
        #output["ids"] = torch.tensor(output["ids"], dtype=torch.long)
        #output["mask"] = torch.tensor(output["mask"], dtype=torch.long)

        return output

class Freeze(Callback):
    def __init__(self, epochs=1, method="hard"):
        self.count = 0
        self.thre = epochs
        self.method = method
        self.done = False
        
        if self.method == "hard":
            logging.info("hard freeze requires_grad -> False")
        else:
            logging.info("soft freeze lr -> 1e-8")
    
    def on_epoch_start(self, model):
        if self.thre == 0:
            self.done = True
            for idx, params in enumerate(model.optimizer_grouped_parameters[2:]):
                logging.info(f"groups {idx + 2} of {len(model.optimizer_grouped_parameters[2:])} has been freeze")
                if self.method == "hard":
                    params = params["params"]
                    for param in params:
                        param.requires_grad = False
                else:
                    params["lr"] = 4e-7
        
    def on_epoch_end(self, model):
        if self.thre == 0:
            return
        
        if self.done:
            return
        
        self.count += 1
        if self.count < self.thre:
            return
        
        logging.info("freeze crf and linear layer")
        self.done = True
        
        for idx, params in enumerate(model.optimizer_grouped_parameters[2:]):
            logging.info(f"groups {idx + 2} of {len(model.optimizer_grouped_parameters[2:])} has been freeze")
            if self.method == "hard":
                params = params["params"]
                for param in params:
                    param.requires_grad = False
            else:
                params["lr"] = 1e-8

class EarlyStopping(Callback):
    def __init__(
        self,
        model_path,
        valid_df,
        valid_samples,
        batch_size,
        tokenizer,
        patience=5,
        mode="max",
        delta=0.001,
        save_weights_only=True,
        direct_output=False,
        bigbird_padding=False
    ):
        self.patience = patience
        self.counter = 0
        self.mode = mode
        self.direct_output = direct_output
        self.best_score = None
        self.best_epoch = None
        self.early_stop = False
        self.delta = delta
        self.save_weights_only = save_weights_only
        self.model_path = model_path
        self.valid_samples = valid_samples
        self.batch_size = batch_size
        self.valid_df = valid_df
        self.tokenizer = tokenizer
        self.epoch = 0
        self.bigbird_padding = bigbird_padding

        if self.mode == "min":
            self.val_score = np.Inf
        else:
            self.val_score = -np.Inf

    def on_epoch_end(self, model):
        model.eval()
        model.save(f"{self.model_path}_epoch{self.epoch}", weights_only=self.save_weights_only)
        valid_dataset = FeedbackDatasetValid(self.valid_samples, 4096, self.tokenizer, bigbird_padding=self.bigbird_padding)
        collate = Collate(self.tokenizer)

        preds_iter = model.predict(
            valid_dataset,
            batch_size=self.batch_size,
            n_jobs=-1,
            collate_fn=collate,
        )

        final_preds = []
        final_scores = []
        for output in preds_iter:
            if self.direct_output:
                pred_class = np.squeeze(output["preds"], axis=0)
                pred_scrs = [[1] * len(_) for _ in pred_class]
            else:
                pred_class = np.argmax(output["preds"], axis=2)
                pred_scrs = np.max(output["preds"], axis=2)
            
            for pred, pred_scr in zip(pred_class, pred_scrs):
                final_preds.append(pred if isinstance(pred, list) else pred.tolist())
                final_scores.append(pred_scr if isinstance(pred_scr, list) else pred_scr.tolist())

        for j in range(len(self.valid_samples)):
            tt = [id_target_map[p] for p in final_preds[j][1:]]
            tt_score = final_scores[j][1:]
            self.valid_samples[j]["preds"] = tt
            self.valid_samples[j]["pred_scores"] = tt_score

        submission = []
        min_thresh = {
            "Lead": 9,
            "Position": 5,
            "Evidence": 14,
            "Claim": 3,
            "Concluding Statement": 11,
            "Counterclaim": 6,
            "Rebuttal": 4,
        }
        proba_thresh = {
            "Lead": 0.7,
            "Position": 0.55,
            "Evidence": 0.65,
            "Claim": 0.55,
            "Concluding Statement": 0.7,
            "Counterclaim": 0.5,
            "Rebuttal": 0.55,
        }

        for _, sample in enumerate(self.valid_samples):
            preds = sample["preds"]
            offset_mapping = sample["offset_mapping"]
            sample_id = sample["id"]
            sample_text = sample["text"]
            sample_pred_scores = sample["pred_scores"]

            # pad preds to same length as offset_mapping
            if len(preds) < len(offset_mapping):
                preds = preds + ["O"] * (len(offset_mapping) - len(preds))
                sample_pred_scores = sample_pred_scores + [0] * (len(offset_mapping) - len(sample_pred_scores))

            idx = 0
            phrase_preds = []
            while idx < len(offset_mapping):
                start, _ = offset_mapping[idx]
                if preds[idx] != "O":
                    label = preds[idx][2:]
                else:
                    label = "O"
                phrase_scores = []
                phrase_scores.append(sample_pred_scores[idx])
                idx += 1
                while idx < len(offset_mapping):
                    if label == "O":
                        matching_label = "O"
                    else:
                        matching_label = f"I-{label}"
                    if preds[idx] == matching_label:
                        _, end = offset_mapping[idx]
                        phrase_scores.append(sample_pred_scores[idx])
                        idx += 1
                    else:
                        break
                if "end" in locals():
                    phrase = sample_text[start:end]
                    phrase_preds.append((phrase, start, end, label, phrase_scores))

            temp_df = []
            for phrase_idx, (phrase, start, end, label, phrase_scores) in enumerate(phrase_preds):
                word_start = len(sample_text[:start].split())
                word_end = word_start + len(sample_text[start:end].split())
                word_end = min(word_end, len(sample_text.split()))
                ps = " ".join([str(x) for x in range(word_start, word_end)])
                if label != "O":
                    if sum(phrase_scores) / len(phrase_scores) >= proba_thresh[label]:
                        temp_df.append((sample_id, label, ps))

            temp_df = pd.DataFrame(temp_df, columns=["id", "class", "predictionstring"])

            submission.append(temp_df)

        submission = pd.concat(submission).reset_index(drop=True)
        submission["len"] = submission.predictionstring.apply(lambda x: len(x.split()))

        def threshold(df):
            df = df.copy()
            for key, value in min_thresh.items():
                index = df.loc[df["class"] == key].query(f"len<{value}").index
                df.drop(index, inplace=True)
            return df

        submission = threshold(submission)

        # drop len
        submission = submission.drop(columns=["len"])
        submission.to_csv(f"{self.model_path}_epoch{self.epoch}.csv")

        scr = score_feedback_comp(submission, self.valid_df, return_class_scores=True)
        logging.info(f"epoch {self.epoch} total:{scr}")
        model.train()

        epoch_score = scr[0]
        if self.mode == "min":
            score = -1.0 * epoch_score
        else:
            score = np.copy(epoch_score)

        if self.best_score is None:
            self.best_score = score
            self.best_epoch = self.epoch
            self.save_checkpoint(epoch_score, model)
            self.save_best_checkpoint(model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            logging.info(f"epoch {self.epoch} EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                model.model_state = enums.ModelState.END
            self.save_checkpoint(epoch_score, model)
        else:
            self.best_score = score
            self.best_epoch = self.epoch
            self.save_checkpoint(epoch_score, model)
            self.save_best_checkpoint(model)
            self.counter = 0
        
        self.epoch += 1

    def save_checkpoint(self, epoch_score, model):
        if epoch_score not in [-np.inf, np.inf, -np.nan, np.nan]:
            logging.info(f"Validation score changed ({self.val_score} --> {epoch_score}). Saving model!")
            logging.info(f"Best performence is epoch {self.best_epoch}: {self.best_score}")
        self.val_score = epoch_score
        
    def save_best_checkpoint(self, model):
        model.save(f"{self.model_path}_best", weights_only=self.save_weights_only)


class GradualWarmupScheduler(_LRScheduler):
    """ Gradually warm-up(increasing) learning rate in optimizer.
    Proposed in 'Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour'.
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        multiplier: target learning rate = base lr * multiplier if multiplier > 1.0. if multiplier = 1.0, lr starts from 0 and ends up with the base_lr.
        total_epoch: target learning rate is reached at total_epoch, gradually
        after_scheduler: after target_epoch, use this scheduler(eg. ReduceLROnPlateau)
    """

    def __init__(self, optimizer, multiplier, warmup_epoch, total_epoch, after_scheduler=None):
        self.multiplier = multiplier
        if self.multiplier < 1.:
            raise ValueError('multiplier should be greater thant or equal to 1.')
        self.warmup_epoch = warmup_epoch
        self.total_epoch = total_epoch
        self.after_scheduler = after_scheduler
        self.finished = False
        super(GradualWarmupScheduler, self).__init__(optimizer)

    def get_lr(self):
        if self.last_epoch % self.total_epoch > self.warmup_epoch:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [base_lr * self.multiplier for base_lr in self.base_lrs]
                    self.finished = True
                return self.after_scheduler.get_last_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]

        if self.multiplier == 1.0:
            return [base_lr * (float(self.last_epoch) / self.warmup_epoch) for base_lr in self.base_lrs]
        else:
            return [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.warmup_epoch + 1.) for base_lr in self.base_lrs]

    def step_ReduceLROnPlateau(self, metrics, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch if epoch != 0 else 1  # ReduceLROnPlateau is called at the end of epoch, whereas others are called at beginning
        self.last_epoch %= self.total_epoch
        if self.last_epoch <= self.warmup_epoch:
            warmup_lr = [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.warmup_epoch + 1.) for base_lr in self.base_lrs]
            for param_group, lr in zip(self.optimizer.param_groups, warmup_lr):
                param_group['lr'] = lr
        else:
            if epoch is None:
                self.after_scheduler.step(metrics, None)
            else:
                self.after_scheduler.step(metrics, self.last_epoch - self.warmup_epoch)

    def step(self, metrics=None, epoch=None):
        if not isinstance(self.after_scheduler, ReduceLROnPlateau):
            if self.finished and self.after_scheduler:
                if epoch is None:
                    self.after_scheduler.step(None)
                else:
                    self.after_scheduler.step(epoch - self.warmup_epoch)
                self._last_lr = self.after_scheduler.get_last_lr()
            else:
                return super(GradualWarmupScheduler, self).step(epoch)
        else:
            self.step_ReduceLROnPlateau(metrics, epoch)

            
class ReduceLROnPlateau(object):
    """Reduce learning rate when a metric has stopped improving.
    Models often benefit from reducing the learning rate by a factor
    of 2-10 once learning stagnates. This scheduler reads a metrics
    quantity and if no improvement is seen for a 'patience' number
    of epochs, the learning rate is reduced.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        mode (str): One of `min`, `max`. In `min` mode, lr will
            be reduced when the quantity monitored has stopped
            decreasing; in `max` mode it will be reduced when the
            quantity monitored has stopped increasing. Default: 'min'.
        factor (float): Factor by which the learning rate will be
            reduced. new_lr = lr * factor. Default: 0.1.
        patience (int): Number of epochs with no improvement after
            which learning rate will be reduced. For example, if
            `patience = 2`, then we will ignore the first 2 epochs
            with no improvement, and will only decrease the LR after the
            3rd epoch if the loss still hasn't improved then.
            Default: 10.
        threshold (float): Threshold for measuring the new optimum,
            to only focus on significant changes. Default: 1e-4.
        threshold_mode (str): One of `rel`, `abs`. In `rel` mode,
            dynamic_threshold = best * ( 1 + threshold ) in 'max'
            mode or best * ( 1 - threshold ) in `min` mode.
            In `abs` mode, dynamic_threshold = best + threshold in
            `max` mode or best - threshold in `min` mode. Default: 'rel'.
        cooldown (int): Number of epochs to wait before resuming
            normal operation after lr has been reduced. Default: 0.
        min_lr (float or list): A scalar or a list of scalars. A
            lower bound on the learning rate of all param groups
            or each group respectively. Default: 0.
        eps (float): Minimal decay applied to lr. If the difference
            between new and old lr is smaller than eps, the update is
            ignored. Default: 1e-8.
        verbose (bool): If ``True``, prints a message to stdout for
            each update. Default: ``False``.

    Example:
        >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        >>> scheduler = ReduceLROnPlateau(optimizer, 'min')
        >>> for epoch in range(10):
        >>>     train(...)
        >>>     val_loss = validate(...)
        >>>     # Note that step should be called after validate()
        >>>     scheduler.step(val_loss)
    """

    def __init__(self, optimizer, mode='min', factor=0.1, patience=10,
                 threshold=1e-4, threshold_mode='rel', cooldown=0,
                 min_lr=0, eps=1e-8, verbose=False):

        if factor >= 1.0:
            raise ValueError('Factor should be < 1.0.')
        self.factor = factor

        # Attach optimizer
        if not isinstance(optimizer, Optimizer):
            raise TypeError('{} is not an Optimizer'.format(
                type(optimizer).__name__))
        self.optimizer = optimizer

        if isinstance(min_lr, list) or isinstance(min_lr, tuple):
            if len(min_lr) != len(optimizer.param_groups):
                raise ValueError("expected {} min_lrs, got {}".format(
                    len(optimizer.param_groups), len(min_lr)))
            self.min_lrs = list(min_lr)
        else:
            self.min_lrs = [min_lr] * len(optimizer.param_groups)

        self.patience = patience
        self.verbose = verbose
        self.cooldown = cooldown
        self.cooldown_counter = 0
        self.mode = mode
        self.threshold = threshold
        self.threshold_mode = threshold_mode
        self.best = None
        self.num_bad_epochs = None
        self.mode_worse = None  # the worse value for the chosen mode
        self.eps = eps
        self.last_epoch = 0
        self._init_is_better(mode=mode, threshold=threshold,
                             threshold_mode=threshold_mode)
        self._reset()

    def _reset(self):
        """Resets num_bad_epochs counter and cooldown counter."""
        self.best = self.mode_worse
        self.cooldown_counter = 0
        self.num_bad_epochs = 0

    def step(self, metrics, epoch=None):
        # convert `metrics` to float, in case it's a zero-dim Tensor
        current = float(metrics)
        if epoch is None:
            epoch = self.last_epoch + 1
        
        self.last_epoch = epoch

        if self.is_better(current, self.best):
            self.best = current
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1

        if self.in_cooldown:
            self.cooldown_counter -= 1
            self.num_bad_epochs = 0  # ignore any bad epochs in cooldown
        if self.num_bad_epochs > self.patience:
            self._reduce_lr(epoch)
            self.cooldown_counter = self.cooldown
            self.num_bad_epochs = 0
        
        self._last_lr = [group['lr'] for group in self.optimizer.param_groups]

    def _reduce_lr(self, epoch):
        for i, param_group in enumerate(self.optimizer.param_groups):
            old_lr = float(param_group['lr'])
            new_lr = max(old_lr * self.factor, self.min_lrs[i])
            if old_lr - new_lr > self.eps:
                param_group['lr'] = new_lr
                if self.verbose:
                    logging.info('Epoch {:5d}: reducing learning rate'
                          ' of group {} to {:.4e}.'.format(epoch, i, new_lr))

    @property
    def in_cooldown(self):
        return self.cooldown_counter > 0

    def is_better(self, a, best):
        if self.mode == 'min' and self.threshold_mode == 'rel':
            rel_epsilon = 1. - self.threshold
            return a < best * rel_epsilon

        elif self.mode == 'min' and self.threshold_mode == 'abs':
            return a < best - self.threshold

        elif self.mode == 'max' and self.threshold_mode == 'rel':
            rel_epsilon = self.threshold + 1.
            return a > best * rel_epsilon

        else:  # mode == 'max' and epsilon_mode == 'abs':
            return a > best + self.threshold

    def _init_is_better(self, mode, threshold, threshold_mode):
        if mode not in {'min', 'max'}:
            raise ValueError('mode ' + mode + ' is unknown!')
        if threshold_mode not in {'rel', 'abs'}:
            raise ValueError('threshold mode ' + threshold_mode + ' is unknown!')

        if mode == 'min':
            self.mode_worse = inf
        else:  # mode == 'max':
            self.mode_worse = -inf

        self.mode = mode
        self.threshold = threshold
        self.threshold_mode = threshold_mode

    def state_dict(self):
        return {key: value for key, value in self.__dict__.items() if key != 'optimizer'}

    def load_state_dict(self, state_dict):
        self.__dict__.update(state_dict)
        self._init_is_better(mode=self.mode, threshold=self.threshold, threshold_mode=self.threshold_mode)

         
