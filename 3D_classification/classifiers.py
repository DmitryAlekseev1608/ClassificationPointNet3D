import torch
import PointNet3D
import train
from torch.utils.data import DataLoader

pointcloud = PointSampler(3000)((verts, faces))
norm_pointcloud = Normalize()(pointcloud)

torch.cuda.empty_cache()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_transforms = transforms.Compose([
                    PointSampler(1024),
                    Normalize(),
                    RandRotation_z(),
                    RandomNoise(),
                    ToTensor()
                    ])

train_ds = PointCloudData(path, transform=train_transforms)
valid_ds = PointCloudData(path, valid=True, folder='test', transform=train_transforms)
inv_classes = {i: cat for cat, i in train_ds.classes.items()}

train_loader = DataLoader(dataset=train_ds, batch_size=8, shuffle=True)
valid_loader = DataLoader(dataset=valid_ds, batch_size=8)

pointnet = PointNet()
pointnet.to(device)
optimizer = torch.optim.Adam(pointnet.parameters(), lr=0.00025)

train.train(pointnet, train_loader, valid_loader)