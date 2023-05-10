from models import MLPModel
import os
from os.path import isfile
## PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim
# Torchvision
import torchvision
from torchvision.datasets import CIFAR10
from torchvision import transforms
# Torch Geometric
import torch_geometric
import torch_geometric.nn as geom_nn
import torch_geometric.data as geom_data

DATASET_PATH = "../data"
project_path = os.path.abspath(os.path.join(os.getcwd(), ".."))
model_path = os.path.join(project_path,"src")
save_file = os.path.join(model_path,"output/mlp/model.p") 

cora_dataset = torch_geometric.datasets.Planetoid(root=DATASET_PATH, name="Cora")

def train(model, dataset, save_file, save_every=100):
    x = dataset.x
    y = dataset.y
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=2e-3)
    loss_function = nn.CrossEntropyLoss()

    EPOCHS = 200
    properties = {}
    for epoch in range(EPOCHS):
        predict = model(x)
        loss = loss_function(predict, y)
        acc = (predict.argmax(dim=-1) == y).sum() / len(y)
        # print(f"epoch= {epoch}, loss= {loss:4.2f}, accuracy= {100.0 * acc: 4.2f}%")
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    # properties.update({'model_state_dict': model.state_dict()})
    # torch.save(save_file, properties)
    torch.save(model.state_dict(), save_file)

    return loss.item(), acc.item()

if __name__ == "__main__":
    
    num_node, c_in = cora_dataset.x.shape
    num_classes = torch.max(cora_dataset.y) + 1 # num

    print(num_node, c_in)
    model = MLPModel(c_in=c_in, c_out=num_classes, c_hidden=16)
    _, train_acc = train(model, dataset = cora_dataset, save_file=save_file)
    print(f"train accuracy = {100 * train_acc:4.2f}%")