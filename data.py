import torchvision
from torchvision import transforms
import torch


def create_dataloader(config):
    preprocess = transforms.Compose(
        [
            transforms.Resize((config.image_size, config.image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    def transform(examples):
        return preprocess(examples.convert("RGB"))


    dataset = torchvision.datasets.MNIST("",train=True,transform=transform,download=True)
    train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=config.train_batch_size, shuffle=True)
    config.training_steps = len(train_dataloader) * config.num_epochs
    return train_dataloader