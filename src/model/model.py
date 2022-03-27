from transformers import AutoModel, AutoConfig
from utils import Freeze
from pytorchcrf import CRF
from loss.sce import SCELoss
from sklearn.metrics import f1_score
from utils import EarlyStopping, prepare_training_data, target_id_map, id_target_map, span_target_id_map, span_id_target_map, GradualWarmupScheduler, ReduceLROnPlateau, span_decode


import pytorch_lightning as pl
import torch.nn as nn
import bitsandbytes as bnb
import numpy as np
import torch


class NBMEModel(pl.LightningModule):
    def __init__(
        self,
        model_name,
        num_train_steps,
        transformer_learning_rate,
        num_labels,
        span_num_labels,
        steps_per_epoch,
        dynamic_merge_layers,
        loss="ce",
        sce_alpha=1.0,
        sce_beta=1.0,
        label_smooth=0.0,
        decoder="softmax",
        max_len=4096,
        merge_layers_num=-2,
        warmup_ratio=0.05,
        finetune=False,
        gradient_ckpt=False,
        max_position_embeddings=None
    ):
        super().__init__()
        self.save_hyperparameters()

        self.cur_step = 0
        self.max_len = max_len
        self.transformer_learning_rate = transformer_learning_rate
        self.dynamic_merge_layers = dynamic_merge_layers
        self.merge_layers_num = merge_layers_num
        self.model_name = model_name
        self.num_train_steps = num_train_steps
        self.num_labels = num_labels
        self.span_num_labels = span_num_labels
        self.label_smooth = label_smooth
        self.decoder = decoder
        self.warmup_ratio = warmup_ratio
        self.finetune = finetune

        hidden_dropout_prob: float = 0.1
        layer_norm_eps: float = 1e-7


        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)

        config.update(
            {
                "output_hidden_states": True,
                "hidden_dropout_prob": hidden_dropout_prob,
                "layer_norm_eps": layer_norm_eps,
                "add_pooling_layer": False,
                "num_labels": self.num_labels,
                
            }
        )
        
        self.transformer = AutoModel.from_pretrained(model_name, config=config)
        if gradient_ckpt:
            self.transformer.gradient_checkpointing_enable()
            
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.2)
        self.dropout3 = nn.Dropout(0.3)
        self.dropout4 = nn.Dropout(0.4)
        self.dropout5 = nn.Dropout(0.5)
        
        
        if self.dynamic_merge_layers:
            self.layer_logits = nn.Linear(config.hidden_size, 1)
        
        if self.decoder == "span":
            self.start_fc = nn.Linear(config.hidden_size, span_num_labels)
            self.end_fc = nn.Linear(config.hidden_size, span_num_labels)
        else:
            self.output = nn.Linear(config.hidden_size, self.num_labels)
            if self.decoder == "crf":
                self.crf = CRF(num_tags=num_labels, batch_first=True)
        
        if loss == "ce":
            #self.loss_layer = nn.CrossEntropyLoss(label_smoothing=label_smooth)
            self.loss_layer = nn.BCEWithLogitsLoss(reduction="none")
        elif loss == "sce":
            self.loss_layer = SCELoss(sce_alpha, sce_beta, num_classes=num_labels if self.decoder != "span" else span_num_labels, label_smooth=label_smooth)
        else:
            raise ValueError("loss set error, must in [ce, sce]")
            
            
    def configure_optimizers(self):
        param_optimizer = list(self.named_parameters())
        no_decay = ["bias", "LayerNorm.bias"]

        transformer_param_optimizer = []
        crf_param_optimizer = []
        other_param_optimizer = []

        for name, para in param_optimizer:
            space = name.split('.')
            if space[0] == 'transformer':
                transformer_param_optimizer.append((name, para))
            elif space[0] == 'crf':
                crf_param_optimizer.append((name, para))
            else:
                other_param_optimizer.append((name, para))
                
        other_lr = self.transformer_learning_rate * 100
        
        self.optimizer_grouped_parameters = [
            {"params": [p for n, p in transformer_param_optimizer if not any(nd in n for nd in no_decay) and p.requires_grad],
             "weight_decay": 0.01, 'lr': self.transformer_learning_rate},
            {"params": [p for n, p in transformer_param_optimizer if any(nd in n for nd in no_decay) and p.requires_grad],
             "weight_decay": 0.0, 'lr': self.transformer_learning_rate},
            
            {"params": [p for n, p in crf_param_optimizer if not any(nd in n for nd in no_decay) and p.requires_grad],
             "weight_decay": 0.01, 'lr': other_lr},
            {"params": [p for n, p in crf_param_optimizer if any(nd in n for nd in no_decay) and p.requires_grad],
             "weight_decay": 0.0, 'lr': other_lr},

            # 其他模块，差分学习率
            {"params": [p for n, p in other_param_optimizer if not any(nd in n for nd in no_decay) and p.requires_grad],
             "weight_decay": 0.01, 'lr': other_lr},
            {"params": [p for n, p in other_param_optimizer if any(nd in n for nd in no_decay) and p.requires_grad],
             "weight_decay": 0.0, 'lr': other_lr},
        ]
        
        opt = bnb.optim.AdamW8bit(self.optimizer_grouped_parameters, lr=self.transformer_learning_rate)

        for module in self.modules():
            if isinstance(module, nn.Embedding):
                bnb.optim.GlobalOptimManager.get_instance().register_module_override(
                    module, 'weight', {'optim_bits': 32}
                )            

        if not self.finetune:
            min_lr = [1e-5, 1e-5, 1e-8, 1e-8, 1e-7, 1e-7]
            patience = 10

            sch = GradualWarmupScheduler(
                opt,
                multiplier=1.1,
                warmup_epoch=int(self.warmup_ratio * self.num_train_steps) ,
                total_epoch=self.num_train_steps)
            
            return [opt], [sch]

        return opt
    
    def loss(self, outputs, targets, attention_mask):
        attention_mask = attention_mask.view(-1)
        outputs = outputs.view(-1, self.num_labels)[attention_mask]
        targets = targets.view(-1, self.num_labels)[attention_mask]

        loss = self.loss_layer(outputs, targets)
        loss = torch.masked_select(loss, targets != -1).mean()
        return loss

    def monitor_metrics(self, outputs, targets, attention_masks, token_type_ids):
        outputs = torch.squeeze(outputs, dim=-1)
        outputs[outputs < 0.5] = 0
        outputs[outputs > 0.5] = 1
        outputs = outputs.long()
        targets = targets.long()

        outputs = torch.masked_select(outputs, attention_masks)
        targets = torch.masked_select(targets, attention_masks)

        mask = targets != -1
        outputs = torch.masked_select(outputs, mask).cpu().detach().numpy()
        targets = torch.masked_select(targets, mask).cpu().detach().numpy()

        f1 = f1_score(outputs, targets)
        return {
                "f1": f1,
                "outputs": outputs,
                "targets": targets
                }

    def forward(self, input_ids, attention_mask, token_type_ids=None, targets=None):
        if token_type_ids is not None:
            transformer_out = self.transformer(input_ids, attention_mask, token_type_ids, output_hidden_states=self.dynamic_merge_layers)
        else:
            transformer_out = self.transformer(input_ids, attention_mask, output_hidden_states=self.dynamic_merge_layers)
            
        if self.dynamic_merge_layers:
            layers_output = torch.cat([torch.unsqueeze(layer, 2) for layer in transformer_out.hidden_states[self.merge_layers_num:]], dim=2)
            layers_logits = self.layer_logits(layers_output)
            layers_weights = torch.transpose(torch.softmax(layers_logits, dim=-1), 2, 3)
            sequence_output = torch.squeeze(torch.matmul(layers_weights, layers_output), 2)
        else:
            sequence_output = transformer_out.last_hidden_state
            
        sequence_output = self.dropout(sequence_output)
        
        logits1 = self.output(self.dropout1(sequence_output))
        logits2 = self.output(self.dropout2(sequence_output))
        logits3 = self.output(self.dropout3(sequence_output))
        logits4 = self.output(self.dropout4(sequence_output))
        logits5 = self.output(self.dropout5(sequence_output))
        logits = self.output(sequence_output)

        probs = torch.sigmoid(logits)
        loss = 0
        
        if targets is not None:
            loss1 = self.loss(logits1, targets, attention_mask=attention_mask)
            loss2 = self.loss(logits2, targets, attention_mask=attention_mask)
            loss3 = self.loss(logits3, targets, attention_mask=attention_mask)
            loss4 = self.loss(logits4, targets, attention_mask=attention_mask)
            loss5 = self.loss(logits5, targets, attention_mask=attention_mask)
            loss = (loss1 + loss2 + loss3 + loss4 + loss5) / 5
            
        metric = self.monitor_metrics(probs, targets, attention_masks=attention_mask, token_type_ids=token_type_ids)["f1"]
        
        return {
            "preds": probs,
            "logits": logits,
            "loss": loss,
            "metric": metric
        }

    def training_step(self, batch, batch_idx):
        output = self(**batch)
        loss = output["loss"]
        f1 = output["metric"]["f1"]
        self.log('train/f1', f1, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train/loss', loss, on_step=True, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        output = self(**batch)
        outputs = output["metric"]["outputs"]
        targets = output["metric"]["targets"]
        
        return {
                "outputs": outputs,
                "targets": targets
                }

    def validation_epoch_end(self, outputs) -> None:
        preds = np.concatenate(outputs["outputs"])
        grounds = np.concatenate(outputs["targets"])
        f1 = f1_score(preds, grounds)
        self.log('valid/f1', f1, on_epoch=True)

