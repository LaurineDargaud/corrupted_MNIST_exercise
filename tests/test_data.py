import torch
from tests import _PATH_DATA

import os.path
import pytest


@pytest.mark.skipif(
    not os.path.exists(f"{_PATH_DATA}/processed/train_dataset.pth"),
    reason="Data files not found",
)
def test_load_training_data():
    # load training data
    train_dataset = torch.load(f"{_PATH_DATA}/processed/train_dataset.pth")
    # check length
    assert (
        len(train_dataset) == 40000
    ), "Dataset did not have the correct number of samples"
    # check shape of elements
    train_labels = []
    for image, label in iter(train_dataset):
        assert image.shape == torch.Size(
            [28, 28]
        ), "Expected each sample to have shape [28, 28]"
        train_labels.append(label.item())
    # check that all labels are represented
    assert list(sorted(set(train_labels))) == [
        i for i in range(10)
    ], "All labels must be represented"


@pytest.mark.skipif(
    not os.path.exists(f"{_PATH_DATA}/processed/test_dataset.pth"),
    reason="Data files not found",
)
def test_load_test_data():
    # load test data
    test_dataset = torch.load(f"{_PATH_DATA}/processed/test_dataset.pth")
    # check length
    # check length
    assert (
        len(test_dataset) == 5000
    ), "Dataset did not have the correct number of samples"
    # check shape of elements
    test_labels = []
    for image, label in iter(test_dataset):
        assert image.shape == torch.Size(
            [28, 28]
        ), "Expected each sample to have shape [28, 28]"
        test_labels.append(label.item())
    # check that all labels are represented
    assert list(sorted(set(test_labels))) == [
        i for i in range(10)
    ], "All labels must be represented"
