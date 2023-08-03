""" Uses a trained model to make predictions on images """

import torch
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt

from typing import List, Tuple
from PIL import Image

def pred_and_plot_image(model: torch.nn.Module,
                        image_path: str,
                        class_names: List[str],
                        image_size: Tuple[int, int],
                        transform: torchvision.transforms,
                        device: torch.device):
    img = Image.open(image_path)

    # create a transform is one doesn't exist
    if transform is not None:
        image_transform = transform
    else:
        image_transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transform.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
        ])
    
    # make sure the model is on the target device
    model.to(device)

    model.eval()
    with torch.inference_mode():
        transformed_image = image_transform(img).unsqueeze(dim=0)

        target_img_pred = model(transformed_image.to(device))
    
    target_img_pred_prob = torch.softmax(target_img_pred, dim=1)
    target_img_pred_label = torch.argmax(target_img_pred_prob, dim=1)

    # plot the figure
    plt.figure()
    plt.imshow(img) 
    plt.title(f"Pred: {class_names[target_img_pred_label]} | Prob: {target_img_pred_prob.max():.3f}")
    plt.axis(False)
