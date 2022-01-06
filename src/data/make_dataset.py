# -*- coding: utf-8 -*-
import logging
from pathlib import Path

import click
import numpy as np
import torch
from dotenv import find_dotenv, load_dotenv
from torch.utils.data import TensorDataset

from omegaconf import OmegaConf

@click.command()
@click.argument("input_filepath", type=click.Path(exists=True))
@click.argument("output_filepath", type=click.Path())
def main(input_filepath, output_filepath):

    # loading nb_files in configuration
    config = OmegaConf.load('src/config/data_conf.yaml')
    nb_files = config.nb_files

    """Runs data processing scripts to turn raw data from (../raw) into
    cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info("making final data set from raw data")

    # export TRAIN dataset
    # to tensor
    all_torch_images, all_torch_labels = [], []
    for i in range(nb_files):
        file_path = input_filepath + "/train_{}.npz".format(i)
        np_array = np.load(file_path)
        all_torch_images.append(torch.from_numpy(np_array["images"]))
        all_torch_labels.append(torch.from_numpy(np_array["labels"]))
    torch_images = torch.cat(all_torch_images, 0)
    torch_labels = torch.cat(all_torch_labels, 0)
    # cast type (float and long)
    torch_images, torch_labels = torch_images.type(
        torch.FloatTensor
    ), torch_labels.type(torch.LongTensor)
    # normalize images
    torch_images = torch.nn.functional.normalize(torch_images, p=1.0)
    # save pth dataset
    train_dataset = TensorDataset(torch_images, torch_labels)
    torch.save(train_dataset, output_filepath + "/train_dataset.pth")

    # export TEST dataset
    file_path = input_filepath + "/test.npz"
    np_array = np.load(file_path)
    torch_test_images = torch.from_numpy(np_array["images"])
    torch_test_labels = torch.from_numpy(np_array["labels"])
    # cast type (float and long)
    torch_test_images, torch_test_labels = torch_test_images.type(
        torch.FloatTensor
    ), torch_test_labels.type(torch.LongTensor)
    # normalize images
    torch_test_images = torch.nn.functional.normalize(torch_test_images, p=1.0)
    # save pth dataset
    test_dataset = TensorDataset(torch_test_images, torch_test_labels)
    torch.save(test_dataset, output_filepath + "/test_dataset.pth")


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
