import torch
from torch import nn
import torch.nn.functional as F
import sys
import math
from .utde_components import *
from ..base import BaseFuseTrainer
from ..registry import ModelRegistry
from torch.optim.lr_scheduler import ReduceLROnPlateau
import copy

import warnings
warnings.filterwarnings("ignore")


@ModelRegistry.register('utde')
class UTDE(BaseFuseTrainer):
    def __init__(self,hparams):
        """
        Construct a MulT Cross model.
        """
        super().__init__()
        self.save_hyperparameters(hparams)
        self.task = self.hparams.task
        
        # Set task-specific number of classes
        if self.task == 'phenotype':
            self.output_dim = self.hparams.num_classes
        elif self.task == 'mortality':
            self.output_dim = 1  # Binary classification
        elif self.task == 'los':
            self.output_dim = 7  # LoS has 7 classes (bins 2-8, excluding 0,1)
        else:
            raise ValueError(f"Unsupported task: {self.task}. Only 'mortality', 'phenotype', and 'los' are supported")
        
        self._init_model_components()
        

    def _init_model_components(self):
        self.model = MULTCrossModel(
            ehr_encoder=getattr(self.hparams, 'ehr_encoder', 'lstm'),
            cxr_encoder=getattr(self.hparams, 'cxr_encoder', 'resnet50'),
            input_dim=getattr(self.hparams, 'input_dim', 498),
            ehr_hidden_dim=getattr(self.hparams, 'ehr_hidden_dim', 128),
            ehr_num_layers=getattr(self.hparams, 'ehr_num_layers', 2),
            ehr_bidirectional=getattr(self.hparams, 'ehr_bidirectional', True),
            ehr_n_head=getattr(self.hparams, 'ehr_n_head', 8),
            pretrained=getattr(self.hparams, 'pretrained', True),
            
            embed_dim=self.hparams.embed_dim,
            embed_time=self.hparams.embed_time,
            cross_layers=self.hparams.cross_layers,
            output_dim=self.output_dim,
            tt_max=self.hparams.tt_max,
            device=self.device,
            num_heads=self.hparams.num_heads,
            dropout=self.hparams.dropout
        )
       
    
    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler"""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=getattr(self.hparams, 'lr', 0.0001)
        )

        scheduler = {
            "scheduler": ReduceLROnPlateau(
                optimizer,
                factor=0.5,
                patience=getattr(self.hparams, 'patience', 10),
                mode='min',
                verbose=True
            ),
            "monitor": "loss/validation_epoch",
            "interval": "epoch",
            "frequency": 1
        }

        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    # reg_ts,  last_cxr,  last_cxr_time
    def forward(self, batch):
        """Forward pass adapting HealNet to our multimodal clinical data"""
        # Extract inputs
        reg_ts = batch['ehr_ts']  # [batch_size, seq_len, features]
        # reg_masks=batch['ehr_masks']
        # concatenate reg_ts and reg_masks
        # reg_ts=torch.cat([reg_ts,reg_masks],dim=-1)
        last_cxr = batch['cxr_imgs']  # [batch_size, channels, height, width]
        last_cxr_time=batch['cxr_times']
        # has_cxr = batch['has_cxr']
        y = batch['labels'].squeeze(-1)
        
        
        # Forward pass 
        pred = self.model(reg_ts,last_cxr,last_cxr_time)
        
        # For mortality task, ensure output shape matches labels
        if self.task == 'mortality' and pred.shape[-1] == 1:
            pred = pred.squeeze(-1)  # [B,1] -> [B]
            
        # Calculate loss
        loss = self.classification_loss(pred, y)
        
        # Return output dict compatible with BaseFuseTrainer
        output = {
            'loss': loss,
            'predictions': pred,
            'labels': y
        }
        
        return output

    def training_step(self, batch, batch_idx):
        """Training step"""
        out = self(batch)
        
        # Log loss
        self.log_dict({'train/loss': out['loss'].detach()}, 
                     on_epoch=True, on_step=True, 
                     batch_size=out['labels'].shape[0],
                     sync_dist=True)
                     
        return {"loss": out['loss'], "pred": out['predictions'].detach(), "labels": out['labels'].detach()}
        
    

