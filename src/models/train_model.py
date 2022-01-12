import argparse
import os
import sys

import torch
from model import MyAwesomeModel
from torch.utils.data import DataLoader

from omegaconf import OmegaConf

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger

# import wandb
# wandb.init()


class train_model(object):
    """
    Train model on train dataset
    then generate the plot of training loss VS steps
    """

    def __init__(self):
        # loading training configuration
        config = OmegaConf.load("src/config/training_conf.yaml")
        self.hparams = config.hyperparameters

        self.train()

    def train(self):
        print("Training day and night")
        parser = argparse.ArgumentParser(description="Training arguments")
        parser.add_argument("--lr", default=self.hparams["lr"], type=float)
        # add any additional argument that you want
        parser.add_argument("--nb_epochs", default=self.hparams["nb_epochs"], type=int)
        parser.add_argument("--save_file", default=self.hparams["save_file"])
        parser.add_argument("--criterion", default=self.hparams["criterion"])
        parser.add_argument("--optimizer", default=self.hparams["optimizer"])
        args = parser.parse_args(sys.argv[1:])
        print(args)

        # TODO: Implement training loop here
        # get model
        optimizer = eval(f"torch.optim.{args.optimizer}")
        criterion = eval(f"torch.nn.{args.criterion}()")
        model = MyAwesomeModel(criterion, optimizer, args.lr)

        # logging with wandb
        # wandb.watch(model)

        # get training dataset
        train_dataset = torch.load(
            os.path.abspath(__file__ + "/../../../data/processed/train_dataset.pth")
        )
        trainloader = DataLoader(
            train_dataset, batch_size=self.hparams["batch_size"], shuffle=True
        )

        # set a callback type checkpoint
        checkpoint_callback = ModelCheckpoint(
            monitor="train_loss", dirpath="./models", mode="min", verbose=True
        )
        checkpoint_callback.FILE_EXTENSION = ".pth"

        # set a callback type earlystop
        early_stopping_callback = EarlyStopping(
            monitor="train_loss", patience=3, verbose=True, mode="min"
        )

        # train
        trainer = Trainer(
            default_root_dir=os.getcwd(),
            max_epochs=args.nb_epochs,
            limit_train_batches=0.2,
            callbacks=[checkpoint_callback, early_stopping_callback],
            logger=WandbLogger(project="corrupted_MNIST_exercise-src_models"),
        )
        trainer.fit(model, trainloader)

        # save best model
        best_model = checkpoint_callback.best_model_path
        best_model_filepath = "./models/" + f"{args.save_file}.pth"
        # create 'models' folder if it doesn't exist
        os.makedirs("models/", exist_ok=True)
        print("here", type(torch.load(best_model)))
        torch.save(torch.load(best_model), best_model_filepath)


if __name__ == "__main__":
    train_model()
