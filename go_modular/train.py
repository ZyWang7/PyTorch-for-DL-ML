""" Trains a PyTorch image classification model """

import os
import torch
from torch import nn
from torchvision import transforms
from timeit import default_timer as timer 

import data_setup, engine, model_builder, utils

# Setup hyperparameters
NUM_EPOCHS = 5 
BATCH_SIZE = 32
HIDDEN_UNITS = 10
LEARNING_RATE = 0.001
DEVICE = "cpu"

# Setup directories
train_dir = "data/pizza_steak_sushi/train"
test_dir = "data/pizza_steak_sushi/test"

# Create transforms
data_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

# Create dataloader and get class names
train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(
    train_dir=train_dir,
    test_dir=test_dir,
    transform=data_transform,
    batch_size=BATCH_SIZE
)

# creata a model
model = model_builder.TinyVGG(input_shape=3,
                              hidden_units=HIDDEN_UNITS,
                              output_shape=len(class_names))

# set up loss function and optimizer
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# train the model and set the timer
start_time = timer()
engine.train(model,
             train_dataloader,
             test_dataloader,
             optimizer,
             loss_fn,
             NUM_EPOCHS,
             DEVICE)
end_time = timer()
print(f"[INFO] Total training time: {end_time-start_time:.3f} seconds")

# save the model to file
utils.save_model(model=model,
                 target_dir="models",
                 model_name="05_go_modular_script_tinyvgg_model.pth")
