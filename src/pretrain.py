import hydra


@hydra.main(config_path='config', config_name='pretrain', version_base="1.3")
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
    joslin_data = {x: JoslinData(data_dir=config.pretrain.data.data_dir,
                                 annotations_file="multiclass_labels_" + x + ".csv",
                                 img_dir="joslin_img") for x in ["train", "test"]}
    joslin_dataloaders = {x: DataLoader(joslin_data[x],
                                        batch_size=config.data.batch_size,
                                        shuffle=config.data.shuffle,
                                        num_workers=config.data.num_workers) for x in ["train", "test"]}

    dataset_sizes = {x: len(joslin_data[x]) for x in ["train", "test"]}
    print("Train dataset size:", dataset_sizes["train"])
    print("Test dataset size:", dataset_sizes["test"])

    class_names = joslin_data["train"].classes
    print("Class names:", class_names)

    # model training
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = create_model(config.model.name,
                         pretrained=config.model.pretrained,
                         num_classes=config.model.num_classes)
    model = model.to(device)

    if config.criterion == "cross_entropy":
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
