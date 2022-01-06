import torch.nn.functional as F
from torch import nn

from omegaconf import OmegaConf

class MyAwesomeModel(nn.Module):
    def __init__(self):
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