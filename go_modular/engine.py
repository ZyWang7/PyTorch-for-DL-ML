
""" contains functions for training and testing a PyTorch model """

from typing import Dict, List, Tuple
import torch
from torch import nn
from tqdm.auto import tqdm

# training steps
def train_step(model: nn.Module,
               dataloader: torch.utils.data.DataLoader,
               loss_fn: nn.Module,
               optimizer: torch.optim.Optimizer,
               device: torch.device):
    model.train()

    # setup train loss and train accuracy values
    train_loss, train_acc = 0, 0

    # loop through dataloader batches
    for batch, (X, y) in enumerate(dataloader):
        # send the data into the target device
        X, y = X.to(device), y.to(device)

        y_pred = model(X)
        loss = loss_fn(y_pred, y)
        train_loss += loss.item()   # get the single integer

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # calculate accuracy metric
        pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        train_acc += (pred_class == y).sum().item() / len(y_pred)
    
    # adjust metrics to get average loss and accuracy per batch
    # -> aberage loss and accuracy per epoch across all batches
    train_loss = train_loss / len(dataloader)
    train_acc = train_acc / len(dataloader)

    return train_loss, train_acc


# testing steps
def test_step(model:nn.Module,
              dataloader: torch.utils.data.DataLoader,
              loss_fn: nn.Module,
              device: torch.device):
    test_loss, test_acc = 0, 0

    model.eval()
    with torch.inference_mode():
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)

            test_pred = model(X)

            loss = loss_fn(test_pred, y)
            test_loss += loss.item()

            test_label = test_pred.argmax(dim=1)
            test_acc += (test_label == y).sum().item() / len(test_label)

    # adjust metrics to get average per batch
    test_loss = test_loss / len(dataloader)
    test_acc = test_acc / len(dataloader)

    return test_loss, test_acc


# train function -> put training step and testing step together
def train(model: nn.Module,
          train_dataloader: torch.utils.data.DataLoader,
          test_dataloader: torch.utils.data.DataLoader,
          optimizer: torch.optim.Optimizer,
          loss_fn: nn.Module,
          epochs: int,
          device: torch.device):
    # create empty results dictionary
    results = {"train_loss" : [],
               "train_acc": [],
               "test_loss" : [],
               "test_acc": []}
    
    # loop through train and test loop
    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = train_step(model, train_dataloader, loss_fn, optimizer, device)
        test_loss, test_acc = test_step(model, test_dataloader, loss_fn, device)

        print(
            f"Epoch: {epoch+1} | "
            f"train_loss: {train_loss:.4f} | "
            f"train_acc: {train_acc:.4f} | "
            f"test_loss: {test_loss:.4f} | "
            f"test_acc: {test_acc:.4f}"
        )
        # update results dictionary
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)
    
    return results
