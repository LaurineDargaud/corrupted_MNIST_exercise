from src.models.model import MyAwesomeModel
from tests import _PATH_DATA

import torch
from torch.utils.data import DataLoader

import pytest

from pytorch_lightning import Trainer

def test_input_output_model_shapes():
    " Check for a given input with shape X that the output of the model have shape Y"
    # load model
    model = MyAwesomeModel(torch.nn.NLLLoss(), torch.optim.SGD, 0.1)
    # load test data loader
    test_dataset = torch.load(f'{_PATH_DATA}/processed/test_dataset.pth')
    testloader = DataLoader(test_dataset, batch_size=64, shuffle=True)
    # make a prediction
    test_features, test_labels = next(iter(testloader))
    trainer = Trainer()
    preds = trainer.predict(model, dataloaders=testloader)[0]
    print(preds.shape)
    # check sizes of input features and output label predictions
    N = test_features.shape[0]
    assert test_features.shape == torch.Size([N,28,28])
    assert preds.shape == torch.Size([N,1])

def test_error_on_wrong_shape():
    with pytest.raises(ValueError, match='Expected input to a 3D tensor'):  
        # load model
        model = MyAwesomeModel(torch.nn.NLLLoss(), torch.optim.SGD, 0.1)
        model(torch.randn(1,2))