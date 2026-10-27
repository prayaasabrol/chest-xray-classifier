from torch.utils.data import DataLoader
from src.data.dataset import ChestXrayDataset
from src.data.transforms import get_train_transforms, get_eval_transforms

def get_dataloaders(data_dir="data", batch_size=32, num_workers=0):
    train_ds = ChestXrayDataset(
        csv_path=f"{data_dir}/processed/train.csv",
        images_dir=f"{data_dir}/raw",
        transform=get_train_transforms()
    )
    val_ds = ChestXrayDataset(
        csv_path=f"{data_dir}/processed/val.csv",
        images_dir=f"{data_dir}/raw",
        transform=get_eval_transforms()
    )
    test_ds = ChestXrayDataset(
        csv_path=f"{data_dir}/processed/test.csv",
        images_dir=f"{data_dir}/raw",
        transform=get_eval_transforms()
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader