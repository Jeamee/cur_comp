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
            self.loss_layer = nn.CrossEntropyLoss(label_smoothing=label_smooth)
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
        outputs = torch.softmax(outputs, dim=-1)
        outputs = outputs.view(-1, self.num_labels)[attention_mask]
        targets = targets.view(-1)[attention_mask]

        loss = self.loss_layer(outputs, targets)
        return loss

    def monitor_metrics(self, outputs, targets, attention_masks, token_type_ids):
        f1 = 0
        outputs = torch.argmax(outputs, dim=-1)
        outputs[outputs == 2] = 0
        targets[targets == 2] = 0
        for output, target, attention_mask, token_type_id in zip(outputs, targets, attention_masks, token_type_ids):
            token_type_id = torch.masked_select(token_type_id, attention_mask)
            output = torch.masked_select(output, attention_mask)
            target = torch.masked_select(target, attention_mask)
            output = torch.masked_select(output, token_type_id)
            target = torch.masked_select(target, token_type_id)
            tmp_f1 = f1_score(output.cpu().numpy(), target.cpu().numpy())
            print(output)
            print(target)
            f1 += tmp_f1

        f1 /= len(outputs)

        return {"f1": f1}

    def forward(self, input_ids, attention_mask, token_type_ids=None, targets=None):
        print(input_ids.shape)
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
        
        if self.decoder == "softmax":
            logits1 = self.output(self.dropout1(sequence_output))
            logits2 = self.output(self.dropout2(sequence_output))
            logits3 = self.output(self.dropout3(sequence_output))
            logits4 = self.output(self.dropout4(sequence_output))
            logits5 = self.output(self.dropout5(sequence_output))
            logits = self.output(sequence_output)
        elif self.decoder == "crf":
            sequence_output1 = self.dropout1(sequence_output)
            sequence_output2 = self.dropout2(sequence_output)
            sequence_output3 = self.dropout3(sequence_output)
            sequence_output4 = self.dropout4(sequence_output)
            sequence_output5 = self.dropout5(sequence_output)
            logits1 = self.output(sequence_output1)
            logits2 = self.output(sequence_output2)
            logits3 = self.output(sequence_output3)
            logits4 = self.output(sequence_output4)
            logits5 = self.output(sequence_output5)
            logits = self.output(sequence_output)
        elif self.decoder == "span":
            sequence_output1 = self.dropout1(sequence_output)
            sequence_output2 = self.dropout2(sequence_output)
            sequence_output3 = self.dropout3(sequence_output)
            sequence_output4 = self.dropout4(sequence_output)
            sequence_output5 = self.dropout5(sequence_output)
            
            start_logits1 = self.start_fc(sequence_output1)
            start_logits2 = self.start_fc(sequence_output2)
            start_logits3 = self.start_fc(sequence_output3)
            start_logits4 = self.start_fc(sequence_output4)
            start_logits5 = self.start_fc(sequence_output5)
            start_logits = (start_logits1 + start_logits2 + start_logits3 + start_logits4 + start_logits5) / 5
            
            end_logits1 = self.end_fc(sequence_output1)
            end_logits2 = self.end_fc(sequence_output2)
            end_logits3 = self.end_fc(sequence_output3)
            end_logits4 = self.end_fc(sequence_output4)
            end_logits5 = self.end_fc(sequence_output5)
            end_logits = (end_logits1 + end_logits2 + end_logits3 + end_logits4 + end_logits5) / 5
            
            logits = (start_logits, end_logits)

        probs = None
        if self.decoder == "softmax":
            probs = torch.softmax(logits, dim=-1)
        elif self.decoder == "crf":
            probs = self.crf.decode(emissions=logits, mask=mask.byte())
        elif self.decoder == "span":
            probs = span_decode(start_logits, end_logits)
        else:
            raise ValueException("except decoder in [softmax, crf]")
        loss = 0
        
        if targets is not None:
            if self.decoder == "softmax":
                loss1 = self.loss(logits1, targets, attention_mask=attention_mask)
                loss2 = self.loss(logits2, targets, attention_mask=attention_mask)
                loss3 = self.loss(logits3, targets, attention_mask=attention_mask)
                loss4 = self.loss(logits4, targets, attention_mask=attention_mask)
                loss5 = self.loss(logits5, targets, attention_mask=attention_mask)
                loss = (loss1 + loss2 + loss3 + loss4 + loss5) / 5
            elif self.decoder == "crf":
                targets = targets * mask
                gather_logits = torch.cat([logits1, logits2, logits3, logits4, logits5], dim=0).cuda()
                gather_targets = torch.cat([targets] * 5, dim=0)
                gather_mask = torch.cat([mask] * 5, dim=0)
                loss = -1. * self.crf(emissions=gather_logits, tags=gather_targets, mask=gather_mask.byte(), reduction='mean')
            elif self.decoder == "span":
                targets, start_targets, end_targets = targets
                
                start_loss1 = self.loss(start_logits1, start_targets, attention_mask=mask)
                start_loss2 = self.loss(start_logits2, start_targets, attention_mask=mask)
                start_loss3 = self.loss(start_logits3, start_targets, attention_mask=mask)
                start_loss4 = self.loss(start_logits4, start_targets, attention_mask=mask)
                start_loss5 = self.loss(start_logits5, start_targets, attention_mask=mask)
                start_loss = (start_loss1 + start_loss2 + start_loss3 + start_loss4 + start_loss5) / 5
                
                end_loss1 = self.loss(end_logits1, end_targets, attention_mask=mask)
                end_loss2 = self.loss(end_logits2, end_targets, attention_mask=mask)
                end_loss3 = self.loss(end_logits3, end_targets, attention_mask=mask)
                end_loss4 = self.loss(end_logits4, end_targets, attention_mask=mask)
                end_loss5 = self.loss(end_logits5, end_targets, attention_mask=mask)
                end_loss = (end_loss1 + end_loss2 + end_loss3 + end_loss4 + end_loss5) / 5
                
                loss = start_loss + end_loss
            else:
                raise ValueException("except decoder in [softmax, crf]")
            
        f1 = self.monitor_metrics(probs, targets, attention_masks=attention_mask, token_type_ids=token_type_ids)["f1"]
        
        return {
            "preds": probs,
            "logits": logits,
            "loss": loss,
            "f1": f1
        }

    def training_step(self, batch, batch_idx):
        output = self(**batch)
        loss = output["loss"]
        f1 = output["f1"]
        self.log('train/loss', loss)
        self.log('train/f1', f1, prog_bar=True)
        self.log('train/avg_f1', f1, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train/avg_loss', loss, on_step=True, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        output = self(**batch)
        f1 = output["f1"]
        self.log('valid/f1', f1, on_step=True, on_epoch=True, prog_bar=True)

