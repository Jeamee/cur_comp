from transformers import AutoModel, AutoConfig, AdamW
from utils import Freeze
from pytorchcrf import CRF
from loss.sce import SCELoss
#from sklearn.metrics import f1_score
from torchmetrics.functional import f1_score
from utils import GradualWarmupScheduler, ReduceLROnPlateau, span_decode


import re
import pytorch_lightning as pl
import torch.nn as nn
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
        lr_decay=1.,
        finetune=False,
        gradient_ckpt=False,
        max_position_embeddings=None,
        use_tpu=False
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
        self.lr_decay = lr_decay
        self.use_tpu = use_tpu

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
        self.num_layers = config.num_hidden_layers
        
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

        transformer_param_optimizer = [[] for _ in range(self.num_layers + 1)]
        other_param_optimizer = []

        for name, para in param_optimizer:
            space = name.split('.')
            if space[0] == 'transformer':
                prob_layer_id = re.findall("\d{1,2}", name)
                layer_num = int(prob_layer_id [0]) if prob_layer_id else 0
                transformer_param_optimizer[layer_num].append((name, para))
            else:
                other_param_optimizer.append((name, para))
                
        other_lr = self.transformer_learning_rate * 100
        
        self.optimizer_grouped_parameters = []
        for idx, layer in enumerate(transformer_param_optimizer):
            lr = self.lr_decay ** (self.num_layers - idx) * self.transformer_learning_rate if idx > 0 else self.transformer_learning_rate

            decay_param_dict = {"params": [p for n, p in layer if not any(nd in n for nd in no_decay) and p.requires_grad],
             "weight_decay": 0.01, 'lr': lr}
            no_decay_param_dict = {"params": [p for n, p in layer if any(nd in n for nd in no_decay) and p.requires_grad],
             "weight_decay": 0.0, 'lr': lr}
            self.optimizer_grouped_parameters.extend([decay_param_dict, no_decay_param_dict])

        self.optimizer_grouped_parameters.extend([
            {"params": [p for n, p in other_param_optimizer if not any(nd in n for nd in no_decay) and p.requires_grad],
             "weight_decay": 0.01, 'lr': other_lr},
            {"params": [p for n, p in other_param_optimizer if any(nd in n for nd in no_decay) and p.requires_grad],
             "weight_decay": 0.0, 'lr': other_lr}
        ])
        
        if not self.use_tpu:
            import bitsandbytes as bnb
            opt = bnb.optim.AdamW8bit(self.optimizer_grouped_parameters, lr=self.transformer_learning_rate)

            for module in self.modules():
                if isinstance(module, nn.Embedding):
                    bnb.optim.GlobalOptimManager.get_instance().register_module_override(
                        module, 'weight', {'optim_bits': 32}
                    )
        else:
            opt = AdamW(self.optimizer_grouped_parameters, lr=self.transformer_learning_rate)      

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
        

        outputs = torch.masked_select(outputs, attention_masks)
        targets = torch.masked_select(targets, attention_masks)


        #f1 = f1_score(outputs, targets)
        return {
                #"f1": f1,
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
        
        if self.training:
            loss1 = self.loss(logits1, targets, attention_mask=attention_mask)
            loss2 = self.loss(logits2, targets, attention_mask=attention_mask)
            loss3 = self.loss(logits3, targets, attention_mask=attention_mask)
            loss4 = self.loss(logits4, targets, attention_mask=attention_mask)
            loss5 = self.loss(logits5, targets, attention_mask=attention_mask)
            loss = (loss1 + loss2 + loss3 + loss4 + loss5) / 5
            metric = None
        else:
            metric = self.monitor_metrics(probs, targets, attention_masks=attention_mask, token_type_ids=token_type_ids)
        
        return {
            "preds": probs,
            "logits": logits,
            "loss": loss,
            "metric": metric
        }

    def training_step(self, batch, batch_idx):
        output = self(**batch)
        loss = output["loss"]
        #f1 = output["metric"]["f1"]
        #self.log('train/f1', f1, on_step=True, on_epoch=True, prog_bar=True)
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
        preds = torch.cat([output["outputs"] for output in outputs])
        grounds = torch.cat([output["targets"] for output in outputs])
        preds[preds < 0.5] = 0
        preds[preds > 0.5] = 1
        preds = preds.long()
        grounds = grounds.long()
        mask = grounds != -1
        preds = torch.masked_select(preds, mask)
        grounds = torch.masked_select(grounds, mask)
        f1 = f1_score(preds, grounds, average=None, num_classes=2)[1]
        self.log('valid/f1', f1, on_epoch=True)
