# Imports here
from matplotlib import pyplot as plt
import numpy as np
import torch
import logging
import json
import os

from torch import nn
from torch import optim
from torchvision import datasets, transforms, models
from collections import OrderedDict
from PIL import Image
from tqdm import tqdm
import seaborn as sns

# Configure logging
logging.basicConfig(level=logging.INFO)


# Check torch version and CUDA status if GPU is enabled.
def check_torch_info():
    """
    Check the torch version and CUDA availability.

    Returns:
    torch_info (dict): A dictionary containing the torch version and CUDA availability.
    """

    torch_info = {
        "version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    return torch_info


# Download data
def download_data(data_url):
    """
    Download the data from the given URL.

    Args:
    data_url (str): The URL to download the data from.
    """

    if not os.path.exists("flowers") and not os.path.exists("flower_data.tar.gz"):
        logging.info("Downloading data...")
        os.system(f"wget {data_url}")
        logging.info("Data downloaded successfully.")
    else:
        logging.info("Data already downloaded.")

    if not os.path.exists("flowers") and os.path.exists("flower_data.tar.gz"):
        logging.info("Extracting data...")
        os.system("mkdir flowers && tar -xvf flower_data.tar.gz -C flowers")
        logging.info("Data extracted successfully.")
    else:
        logging.info("Data already extracted.")

    logging.info("Data download and extraction complete.")

    if os.path.exists("flowers") and os.path.exists("flower_data.tar.gz"):
        os.remove("flower_data.tar.gz")


# Init data to set data dirs and data loaders
def init_data(data_dir, batch_size=64, shuffle=True):
    """
    Initialize the data directories and data loaders.

    Args:
    data_dir (str): The data directory path.
    batch_size (int): The batch size for the data loaders.
    shuffle (bool): Whether to shuffle the data or not.

    Returns:
    image_datasets (dict): A dictionary containing the image datasets.
    dataloaders (dict): A dictionary containing the data loaders.
    """
    try:
        train_dir = data_dir + "/train"
        valid_dir = data_dir + "/valid"
        test_dir = data_dir + "/test"

        logging.info("Data directories are now set!")

        # Define the data transforms for the training, validation, and testing sets
        data_transforms = {
            "train": transforms.Compose(
                [
                    transforms.RandomRotation(30),
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ]
            ),
            "valid": transforms.Compose(
                [
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ]
            ),
            "test": transforms.Compose(
                [
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ]
            ),
        }

        # Load the datasets with ImageFolder
        image_datasets = {
            "train": datasets.ImageFolder(
                train_dir, transform=data_transforms["train"]
            ),
            "valid": datasets.ImageFolder(
                valid_dir, transform=data_transforms["valid"]
            ),
            "test": datasets.ImageFolder(test_dir, transform=data_transforms["test"]),
        }

        # Using the image datasets and the trainforms, define the dataloaders
        dataloaders = {
            "train": torch.utils.data.DataLoader(
                image_datasets["train"], batch_size=batch_size, shuffle=shuffle
            ),
            "valid": torch.utils.data.DataLoader(
                image_datasets["valid"], batch_size=batch_size
            ),
            "test": torch.utils.data.DataLoader(
                image_datasets["test"], batch_size=batch_size
            ),
        }

        logging.info("Data loaders are now set!")

        return image_datasets, dataloaders

    except Exception as e:
        logging.error(f"An error occurred while initializing data: {e}")
        return None, None


# Set device to use GPU if available
def set_device(device="gpu"):
    """
    Set the device to use GPU if available.

    Args:
    device (str): The device to use. Default is 'gpu'.

    Returns:
    device (torch.device): The device to use.
    """
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    logging.info(f"Using device: {device}")

    return device


# Define model_setup func to setup model, optimizer and criterion
def model_setup(device, structure="vgg16", hidden_layer=4096, lr=0.001, dropout=0.2):
    """
    Setup the model, optimizer and criterion.

    Args:
    device (torch.device): The device to use.
    structure (str): The model architecture to use. Default is 'vgg16'.
    hidden_layer (int): The number of hidden layers. Default is 4096.
    lr (float): The learning rate for the optimizer. Default is 0.001.
    dropout (float): The dropout rate. Default is 0.2.

    Returns:
    model (torch.nn.Module): The model to use.
    optimizer (torch.optim): The optimizer to use.
    criterion (torch.nn): The criterion to use.
    """
    structures = {"vgg16": 25088, "alexnet": 9216, "resnet18": 512, "densenet121": 1024}

    if structure == "vgg16":
        model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
    elif structure == "alexnet":
        model = models.alexnet(weights=models.AlexNet_Weights.IMAGENET1K_V1)
    elif structure == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    elif structure == "densenet121":
        model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
    else:
        raise ValueError(
            f"The {structure} is not a valid model. Please input either 'vgg16', 'alexnet', 'resnet18', or 'densenet121'!"
        )

    logging.info(f"Using {structure} model with pre-trained weights.")

    for param in model.parameters():
        param.requires_grad = False

    if structure in ["vgg16", "alexnet"]:
        classifier = nn.Sequential(
            OrderedDict(
                [
                    ("dropout1", nn.Dropout(dropout)),
                    ("fc1", nn.Linear(structures[structure], hidden_layer)),
                    ("relu1", nn.ReLU()),
                    ("fc2", nn.Linear(hidden_layer, 1024)),
                    ("relu2", nn.ReLU()),
                    ("fc3", nn.Linear(1024, 102)),
                    ("output", nn.LogSoftmax(dim=1)),
                ]
            )
        )
        model.classifier = classifier
    elif structure == "resnet18":
        model.fc = nn.Sequential(
            OrderedDict(
                [
                    ("dropout1", nn.Dropout(dropout)),
                    ("fc1", nn.Linear(structures[structure], hidden_layer)),
                    ("relu1", nn.ReLU()),
                    ("fc2", nn.Linear(hidden_layer, 102)),
                    ("output", nn.LogSoftmax(dim=1)),
                ]
            )
        )
    elif structure == "densenet121":
        model.classifier = nn.Sequential(
            OrderedDict(
                [
                    ("dropout1", nn.Dropout(dropout)),
                    ("fc1", nn.Linear(structures[structure], hidden_layer)),
                    ("relu1", nn.ReLU()),
                    ("fc2", nn.Linear(hidden_layer, 102)),
                    ("output", nn.LogSoftmax(dim=1)),
                ]
            )
        )

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(
        (
            model.classifier.parameters()
            if structure in ["vgg16", "alexnet", "densenet121"]
            else model.fc.parameters()
        ),
        lr,
    )

    model.to(device)

    logging.info("Model setup complete.")

    return model, optimizer, criterion


# Validate the model with the validation dataset to check the accuracy and loss
def validate(device, model, data_loader, criterion):
    """
    Validate the model with the validation dataset to check the accuracy and loss.

    Args:
    device (torch.device): The device to use.
    model (torch.nn.Module): The model to use.
    data_loader (torch.utils.data.DataLoader): The data loader to use.
    criterion (torch.nn): The criterion to use.

    Returns:
    avg_val_loss (float): The average validation loss.
    avg_val_accuracy (float): The average validation accuracy.
    """

    val_loss = 0
    val_accuracy = 0
    total = 0

    # Turn off gradients for validation, saves memory and computations
    with torch.no_grad():
        model.eval()
        for images, labels in tqdm(data_loader, desc="Validating"):
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)

            loss = criterion(outputs, labels)
            val_loss += loss.item()

            ps = torch.exp(outputs)
            _, indices = ps.topk(1, dim=1)
            equality = indices == labels.view(*indices.shape)
            val_accuracy += torch.sum(equality.type(torch.FloatTensor)).item()
            total += labels.size(0)

    # Calculate the average loss and accuracy
    avg_val_loss = val_loss / len(data_loader)
    avg_val_accuracy = val_accuracy / total

    logging.info(
        f"Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {avg_val_accuracy:.4f}"
    )

    return avg_val_loss, avg_val_accuracy


# Define train_model func
def train_model(
    device,
    dataloaders,
    image_datasets,
    model,
    optimizer,
    criterion,
    epochs=5,
    print_every=50,
):
    """
    Train the model with the training dataset.

    Args:
    device (torch.device): The device to use.
    dataloaders (dict): The data loaders to use.
    image_datasets (dict): The image datasets to use.
    model (torch.nn.Module): The model to use.
    optimizer (torch.optim): The optimizer to use.
    criterion (torch.nn): The criterion to use.
    epochs (int): The number of epochs to train the model. Default is 5.
    print_every (int): The number of steps to print the training loss and validation loss. Default is 50.
    """

    logging.info("Training process is now starting...")
    logging.info(f"Device: {device}")

    if image_datasets is None:
        logging.error("Image datasets is None. Please initialize the image datasets.")
        return

    # Check if dataloaders and image_datasets are not None
    if dataloaders is None:
        logging.error("Dataloaders is None. Please initialize the data loaders.")
        return

    # Check if 'train' and 'valid' keys exist in dataloaders
    if "train" not in dataloaders or "valid" not in dataloaders:
        logging.error("'train' or 'valid' key is missing in dataloaders.")
        return

    model.train()
    steps = 0
    running_loss = 0
    val_len = len(dataloaders["valid"])

    for epoch in range(epochs):
        for images, labels in tqdm(
            dataloaders["train"], desc="Epoch: {}/{}".format(epoch + 1, epochs)
        ):
            steps += 1

            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                val_loss, val_accuracy = validate(
                    device, model, dataloaders["valid"], criterion
                )

                # Log the training loss and validation loss
                logging.info(
                    f"Epoch: {epoch + 1}/{epochs}.."
                    f"Training loss: {running_loss/print_every:.3f}.."
                    f"Validation loss: {val_loss/val_len:.3f}.."
                    f"Validation accuracy: {val_accuracy/val_len:.3f}"
                )
                running_loss = 0
                model.train()
    model.class_to_idx = image_datasets["train"].class_to_idx
    logging.info("Training process is now complete!")


# Save the checkpoint
def save_checkpoint(
    model,
    optimizer,
    structure,
    image_datasets,
    save_dir="checkpoint.pth",
    hidden_layer=4096,
):
    """
    Save the checkpoint of the model.

    Args:
    model (torch.nn.Module): The model to save.
    optimizer (torch.optim): The optimizer to save.
    structure (str): The model architecture to save.
    image_datasets (dict): The image datasets to save.
    save_dir (str): The directory to save the checkpoint. Default is 'checkpoint.pth'.
    hidden_layer (int): The number of hidden layers. Default is 4096.
    """

    model.class_to_idx = image_datasets["train"].class_to_idx
    # Save the checkpoint
    checkpoint = {
        "structure": structure,
        "hidden_layer": hidden_layer,
        "state_dict": model.state_dict(),
        "class_to_idx": model.class_to_idx,
        "optimizer": optimizer.state_dict(),
    }

    torch.save(checkpoint, save_dir)
    logging.info(f"Checkpoint saved to {save_dir}")


# Write a function that loads a checkpoint and rebuilds the model
def load_checkpoint(filepath, device):
    """
    Load the checkpoint and rebuild the model.

    Args:
    filepath (str): The path to the checkpoint.
    device (torch.device): The device to use.

    Returns:
    model (torch.nn.Module): The model to use.
    optimizer (torch.optim): The optimizer to use.
    criterion (torch.nn): The criterion to use.
    """
    checkpoint = torch.load(filepath, weights_only=True)

    structure = checkpoint["structure"]
    model, optimizer, criterion = model_setup(device, structure)

    model.class_to_idx = checkpoint["class_to_idx"]
    model.load_state_dict(checkpoint["state_dict"])

    return model, optimizer, criterion


# Process image
def process_image(image_path):
    """
    Process an image for use in a PyTorch model.

    Args:
    image_path (str): The path to the image.

    Returns:
    img (torch.Tensor): The processed image as a PyTorch tensor.
    """

    # TODO: Process a PIL image for use in a PyTorch model
    img = Image.open(image_path)
    img_transforms = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    img = img_transforms(img)

    return img


def imshow(image, ax=None, title=None):
    """
    Imshow for Tensor.

    Args:
    image (torch.Tensor): The image to display.
    ax (matplotlib.axes.Axes): The axes to display the image.
    title (str): The title of the image.

    Returns:
    ax (matplotlib.axes.Axes): The axes to display the image.
    """
    if ax is None:
        fig, ax = plt.subplots()

    # PyTorch tensors assume the color channel is the first dimension
    image = image.numpy().transpose((1, 2, 0))

    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean

    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)

    # Display the image
    ax.imshow(image)

    return ax


# Predict the class of an image
def predict(image_path, model, device, cat_to_name, topk=5):
    """
    Predict the class of an image.

    Args:
    image_path (str): The path to the image.
    model (torch.nn.Module): The model to use.
    device (torch.device): The device to use.
    cat_to_name (dict): The category to name mapping.
    topk (int): The number of top classes to return. Default is 5.

    Returns:
    probs (list): The probabilities of the top classes.
    classes (list): The classes of the top classes.
    flowers (list): The names of the top classes.
    """
    model.eval()
    img = process_image(image_path)

    # Convert 2D image to 1D vector
    with torch.no_grad():
        # Convert 2D image to 1D vector
        img = img.unsqueeze(0)
        img = img.to(device)

        output = model(img)
        ps = torch.exp(output)
        probs, indices = ps.topk(topk)

        # Convert indices to classes
        probs = probs.cpu().numpy().tolist()[0]
        indices = indices.cpu().numpy().tolist()[0]

        # Invert the class_to_idx dictionary
        idx_to_class = {val: key for key, val in model.class_to_idx.items()}
        classes = [idx_to_class[idx] for idx in indices]
        flowers = [cat_to_name[cls] for cls in classes]

    return probs, classes, flowers


# Load label mapping
def load_label_mapping(category_names):
    """
    Load the label mapping from a JSON file.

    Args:
    category_names (str): The path to the JSON file.

    Returns:
    cat_to_name (dict): The category to name mapping.
    """

    try:
        with open(category_names, "r") as f:
            cat_to_name = json.load(f)
        logging.info("Label mapping is now complete!")
        return cat_to_name
    except Exception as e:
        logging.error(f"Error loading label mapping: {e}")
        return None


# Test the model
def test_model(device, model, criterion, dataloader):
    """
    Test the model with the test dataset.

    Args:
    device (torch.device): The device to use.
    model (torch.nn.Module): The model to use.
    criterion (torch.nn): The criterion to use.
    dataloader (torch.utils.data.DataLoader): The data loader to use.

    Returns:
    test_loss (float): The test loss.
    test_accuracy (float): The test accuracy.
    """
    test_loss = 0
    test_accuracy = 0

    with torch.no_grad():
        model.eval()
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)

            test_loss += criterion(outputs, labels)

            ps = torch.exp(outputs)
            probs, indices = ps.topk(1, dim=1)
            equalilty = indices == labels.view(*indices.shape)
            test_accuracy += torch.mean(equalilty.type(torch.FloatTensor)).item()

    return test_loss, test_accuracy


def display_image(image_path, model, device, cat_to_name):
    """
    Display the image with the prediction.

    Args:
    image_path (str): The path to the image.
    model (torch.nn.Module): The model to use.
    device (torch.device): The device to use.
    cat_to_name (dict): The category to name mapping.
    """

    # Set up the plot
    plt.figure(figsize=(6, 10))
    ax = plt.subplot(2, 1, 1)

    # Set up title
    flower_num = image_path.split("/")[2]
    title = cat_to_name[flower_num]

    # Process image
    img = process_image(image_path)
    imshow(img, ax, title=title)

    # Make prediction
    probs, classes, flowers = predict(image_path, model, device, cat_to_name)

    # Plot bar chart
    plt.subplot(2, 1, 2)
    sns.barplot(x=probs, y=flowers, color=sns.color_palette()[0])
    plt.show()
