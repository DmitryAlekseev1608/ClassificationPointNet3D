import torch
import point_net_3d
import model_net_10
from torch.utils.data import DataLoader, Dataset
import os
from pathlib import Path

path = Path("test")
inv_classes = {
0: 'bathtub',
 1: 'bed',
 2: 'chair',
 3: 'desk',
 4: 'dresser',
 5: 'monitor',
 6: 'night_stand',
 7: 'sofa',
 8: 'table',
 9: 'toilet'}

pointnet = point_net_3d.PointNet()
pointnet.load_state_dict(torch.load('models/save.pth'))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
valid_ds = model_net_10.PointCloudData(path, valid=True, folder='', transform=model_net_10.train_transforms)

valid_loader = DataLoader(dataset=valid_ds, batch_size=8)

pointnet.eval()
all_preds = []
with torch.no_grad():
    for i, data in enumerate(valid_loader):
       
        inputs, labels = data['pointcloud'].float(), data['category']
        outputs, __, __ = pointnet(inputs.transpose(1,2))
        _, preds = torch.max(outputs.data, 1)
        _, preds = torch.max(outputs.data, 1)
        all_preds += list(preds.numpy())

for i in range(len(all_preds)):
    print(inv_classes[all_preds[i]])