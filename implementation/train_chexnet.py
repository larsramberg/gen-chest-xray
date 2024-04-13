from torchvision import transforms
from models.classifiers import chexnet, chexnet_train_one_epoch, fit_chexnet
import torch.optim
from torch.utils.data import DataLoader
from dotenv import load_dotenv
import os
from util.plot_tools import show_and_save_series_plot, save_series_plot
from datahandling.transforms import to_numeric_label, to_class_int
from dataset.chestxray import ChestXRayDataset
from torch.utils.tensorboard import SummaryWriter
from accelerate import Accelerator

### Load environment ###
load_dotenv()
db_path = os.getenv("DB_PATH")
img_dir_name = os.getenv("IMG_DIR")

annotation_folder = os.path.join(db_path, "training_sets/chexnet")
img_dir = os.path.join(db_path, img_dir_name)
accelerator = Accelerator()

### Load data ###
device = "cuda"

transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

batch_size = 16
num_workers = 6

train_set = ChestXRayDataset(os.path.join(annotation_folder, "train_gh.csv"), img_dir, read_lib="pil", transform=transform, target_transform=to_numeric_label)
val_set = ChestXRayDataset(os.path.join(annotation_folder, "validation_gh.csv"), img_dir, read_lib="pil", transform=transform, target_transform=to_numeric_label)
test_set = ChestXRayDataset(os.path.join(annotation_folder, "test_gh.csv"), img_dir, read_lib="pil", transform=transform, target_transform=to_numeric_label)

train_loader = DataLoader(train_set, batch_size, pin_memory=True, shuffle=True, num_workers=num_workers)
val_loader = DataLoader(val_set, batch_size, pin_memory=True, num_workers=num_workers)
test_set = DataLoader(test_set, batch_size, pin_memory=True, num_workers=num_workers)

### Pick Model ###
model = chexnet(len(ChestXRayDataset.target_labels), device)
print("Training on device:", torch.cuda.get_device_name(device))

### Pick optimizer, loss function, and set scheduler. Train model ###
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.999), weight_decay= 1e-5)
lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=1, threshold=1e-10)
loss_fn = torch.nn.BCELoss(reduction="mean")
num_epochs = 100

avg_train_loss_series, avg_val_loss_series, avg_val_acc_series = fit_chexnet(
    model, 
    optimizer, 
    lr_scheduler, 
    loss_fn, 
    train_loader, 
    val_loader, 
    num_epochs,
    len(ChestXRayDataset.target_labels),
    "result/chexnet/weights",
    device
    )

# Save Graph of loss and accuracy series
result_folder = "result/chexnet/graphs"
save_series_plot(avg_train_loss_series, "training loss", result_folder, "chexnet_train_loss", True)
save_series_plot(avg_val_loss_series, "validation loss", result_folder, "chexnet_val_loss", True)
save_series_plot(avg_val_acc_series, "validation accuracy", result_folder, "chexnet_val_acc", True)