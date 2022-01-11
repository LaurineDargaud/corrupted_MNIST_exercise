from pytorch_lightning import LightningModule

import torch.nn.functional as F
from torch import nn, optim

from omegaconf import OmegaConf

import wandb

class MyAwesomeModel(LightningModule):
    def __init__(self, aCriterium, anOptimizer, aLrCoeff):
        super().__init__()
        # get hyperparameters
        config = OmegaConf.load('src/config/model_conf.yaml')
        hparams = config.hyperparameters
        self.fc1 = nn.Linear(hparams['input_size'], hparams['layer1_size'])
        # self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(hparams['layer1_size'], hparams['layer2_size'])
        self.fc4 = nn.Linear(hparams['layer2_size'], hparams['output_size'])

        # Dropout module with 0.2 drop probability
        self.dropout = nn.Dropout(p=hparams['p_dropout'])

        # Define criterium
        self.criterium = aCriterium

        # Define optimizer and lr
        self.optimizer = anOptimizer
        self.lr = aLrCoeff

    def forward(self, x):
        # make sure input tensor is flattened
        x = x.view(x.shape[0], -1)

        # now with dropout
        x = self.dropout(F.relu(self.fc1(x)))
        # x = self.dropout(F.relu(self.fc2(x)))
        x = self.dropout(F.relu(self.fc3(x)))

        # output without dropout
        x = F.log_softmax(self.fc4(x), dim=1)

        return x

    def get_intermediate_outputs(self, x):
        """
        Return intermediate outputs after the first and second layer processes

            Parameters:
                x (float tensor) : Training data

            Returns:
                x1, x2 (float tensor) : First and Second layer outputs
        """
        # make sure input tensor is flattened
        x = x.view(x.shape[0], -1)

        # now with dropout
        x1 = self.dropout(F.relu(self.fc1(x)))
        # x = self.dropout(F.relu(self.fc2(x)))
        x2 = self.dropout(F.relu(self.fc3(x1)))

        return x1, x2
    
    def _shared_eval_step(self, batch, batch_idx):
        data, target = batch
        preds = self(data)
        loss = self.criterium(preds, target)
        acc = (target == preds.argmax(dim=-1)).float().mean()
        return loss, acc
    
    def training_step(self, batch, batch_idx):
        data, target = batch
        preds = self(data)
        loss = self.criterium(preds, target)
        return loss
    
    def configure_optimizers(self):
        return self.optimizer(self.parameters(), lr=self.lr)
    
    def validation_step(self, batch, batch_idx):
        # compute and log validation metrics
        loss, acc = self._shared_eval_step(batch, batch_idx)
        metrics = {"val_acc":acc, "val_loss":loss}
        self.log_dict(metrics)
        #  logging something else than scalar tensors
        preds = self.predict_step(batch, batch_idx)
        self.logger.experiment.log({'logits': wandb.Histogram(preds.detach().numpy())})
        return metrics
    
    def test_step(self, batch, batch_idx):
        # compute and log test metrics
        loss, acc = self._shared_eval_step(batch, batch_idx)
        metrics = {"test_acc":acc, "test_loss":loss}
        self.log_dict(metrics)
        return metrics
    
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        data, _ = batch
        preds = self(data)
        return preds
