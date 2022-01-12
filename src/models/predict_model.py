import argparse
import os
import sys

import torch
from torch.utils.data import DataLoader

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger


class predict_model(object):
    """Test a trained model on a test dataset then display obtained accuracy"""

    def __init__(self):
        self.predict()

    def predict(self):
        print("Evaluating until hitting the ceiling")
        parser = argparse.ArgumentParser(description="Training arguments")
        parser.add_argument("load_model_from", default="")
        # add any additional argument that you want
        parser.add_argument("test_dataset", default="")
        args = parser.parse_args(sys.argv[1:])
        print(args)

        # TODO: Implement evaluation logic here

        # get trained model
        model = torch.load(
            os.path.abspath(__file__ + "/../../../" + args.load_model_from)
        )
        # get test dataloader
        test_dataset = torch.load(
            os.path.abspath(__file__ + "/../../../" + args.test_dataset)
        )
        testloader = DataLoader(test_dataset, batch_size=64, shuffle=True)

        print("here", args.load_model_from)

        # get trainer
        trainer = Trainer(
            default_root_dir=os.getcwd(),
            logger=WandbLogger(project="corrupted_MNIST_exercise-src_models"),
        )
        trainer.test(model, dataloaders=testloader)


if __name__ == "__main__":
    predict_model()
