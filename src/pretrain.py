import hydra
from sklearn.model_selection import train_test_split

# @hydra.main(config_path='conf', config_name='pretrain')
@hydra.main(config_path='/content/drive/MyDrive/RETProgression/config', config_name='pretrain', version_base=None)
def run(config):
    # deferred imports for faster tab completion
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.optim import lr_scheduler
    from torch.utils.data import DataLoader

    from timm import create_model

    from datasets import JoslinData
    from utils import train_model

    # load datasets
    # joslin_data = {x: JoslinData(data_dir=config.pretrain.data.data_dir,
    #                              annotations_file="labels_" + x + ".csv",
    #                              img_dir="images") for x in ["train", "val"]}
    # joslin_dataloaders = {x: DataLoader(joslin_data[x],
    #                                     batch_size=config.data.batch_size,
    #                                     shuffle=config.data.shuffle,
    #                                     num_workers=config.data.num_workers) for x in ["train", "val"]}

    # load datasets
    # joslin_data = {x: JoslinData(bucket_name=bucket_name,
    #                              annotations_file_path=annotations_file_path,
    #                              transform=None)  # Add necessary transforms here
    #                 for x in ["train", "val"]}
    # print(joslin_data)                
    # joslin_dataloaders = {x: DataLoader(joslin_data[x],
    #                                     batch_size=config.data.batch_size,
    #                                     shuffle=config.data.shuffle,
    #                                     num_workers=config.data.num_workers)
    #                       for x in ["train", "val"]}

    #gcp specific start 
    # GCP bucket name and Colab path to the labels file
    bucket_name = 'arvo_2022_images'
    annotations_file_path = "/content/drive/My Drive/csv_NG_IMAGES_GRADES.csv"  # Update this path to where your labels file is stored in Colab

    # Load the full dataset to get indices
    full_dataset = JoslinData(bucket_name=bucket_name, annotations_file_path=annotations_file_path)
    
    # Split the dataset into 70% training and 30% validation
    train_idx, val_idx = train_test_split(range(len(full_dataset)), test_size=0.3, random_state=42)
    
    # Create separate datasets for training and validation using the split indices
    joslin_data = {
        'train': JoslinData(bucket_name=bucket_name, annotations_file_path=annotations_file_path, indices=train_idx),
        'val': JoslinData(bucket_name=bucket_name, annotations_file_path=annotations_file_path, indices=val_idx)
    }

    # Create dataloaders for training and validation
    joslin_dataloaders = {
        'train': DataLoader(joslin_data['train'], batch_size=config.data.batch_size, shuffle=config.data.shuffle, num_workers=config.data.num_workers, pin_memory=False, persistent_workers=False),
        'val': DataLoader(joslin_data['val'], batch_size=config.data.batch_size, shuffle=False, num_workers=config.data.num_workers, pin_memory=False, persistent_workers=False)
    }
    
    #gcp specific end

    dataset_sizes = {x: len(joslin_data[x]) for x in ['train', 'val']}
    print("Train dataset size:", dataset_sizes["train"])
    print("Validation dataset size:", dataset_sizes["val"])

    class_names = joslin_data["train"].classes
    print("Class names:", class_names)

    # model training
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = create_model(config.model.name,
                         pretrained=config.model.pretrained,
                         num_classes=config.model.num_classes)
    model = model.to(device)

    #if config.criterion == "cross_entropy":
    criterion = nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized
    if config.optimizer.name == "adam":
        optimizer_ft = optim.Adam(model.parameters(), lr=config.optimizer.lr)
    elif config.optimizer.name == "sgd":
        optimizer_ft = optim.SGD(model.parameters(),
                                 lr=config.optimizer.lr,
                                 momentum=config.optimizer.momentum)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft,
                                           step_size=config.lr_scheduler.step_size,
                                           gamma=config.lr_scheduler.gamma)

    model = train_model(joslin_dataloaders, dataset_sizes,
                        model, criterion, optimizer_ft,
                        exp_lr_scheduler, device,
                        num_epochs=config.exp.num_epochs)


if __name__ == '__main__':
    run()
