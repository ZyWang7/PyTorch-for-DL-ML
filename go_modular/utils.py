""" cintains various utility functions for PyTorvh model training """

import torch
from torch import nn
from pathlib import Path

def save_model(model: nn.Module, target_dir: str, model_name: str):
    """ save a model yo a target directory """
    # Create target directory
    target_dir_path = Path(target_dir)
    target_dir_path.mkdir(parents=True, exist_ok=True)

    # create the model save path
    assert model_name.endswith(".pth") or model_name.endswith(".pt"), "model_name should end with '.pt' or '.pth'"
    model_save_path = target_dir_path / model_name

    print("[INFO] Saving model to:", model_save_path)
    torch.save(obj=model.state_dict(), f=model_save_path)
