import argparse
import logging
from utils import init_data, set_device, model_setup, train_model, save_checkpoint
import json

# Configure logging
logging.basicConfig(level=logging.INFO)


def main():
    arg_parser = argparse.ArgumentParser(description="train.py")

    arg_parser.add_argument(
        "data_dir", nargs="?", action="store", default="flowers", type=str
    )
    arg_parser.add_argument(
        "--save_dir",
        dest="save_dir",
        action="store",
        default="checkpoint.pth",
        type=str,
    )
    arg_parser.add_argument(
        "--arch", dest="arch", action="store", default="vgg16", type=str
    )
    arg_parser.add_argument(
        "--learning_rate", dest="learning_rate", action="store", default=0.001
    )
    arg_parser.add_argument(
        "--hidden_units", dest="hidden_units", action="store", default=4096, type=int
    )
    arg_parser.add_argument(
        "--epochs", dest="epochs", action="store", default=5, type=int
    )
    arg_parser.add_argument(
        "--gpu", dest="gpu", action="store", default="gpu", type=str
    )
    arg_parser.add_argument(
        "--config",
        dest="config",
        action="store",
        default=None,
        type=str,
        help="Path to configuration file",
    )
    arg_parser.add_argument(
        "--batch_size", dest="batch_size", action="store", default=64, type=int
    )
    params = arg_parser.parse_args()

    # Load parameters from configuration file if provided
    if params.config:
        with open(params.config, "r") as f:
            config_params = json.load(f)
            params.__dict__.update(config_params)

    try:
        device = set_device(params.gpu)

        # Ensure data_dir is a string
        if not isinstance(params.data_dir, str):
            raise ValueError("data_dir must be a string")

        logging.info("Data Directory: " + params.data_dir)

        image_datasets, dataloaders = init_data(
            params.data_dir, params.batch_size, shuffle=True
        )
        logging.info("Data loaded successfully.")

        model, optimizer, criterion = model_setup(
            device, params.arch, params.hidden_units, params.learning_rate
        )
        logging.info(f"Model setup complete with architecture: {params.arch}")

        train_model(
            device,
            dataloaders,
            image_datasets,
            model,
            optimizer,
            criterion,
            params.epochs,
        )

        save_checkpoint(model, optimizer, params.arch, image_datasets, params.save_dir)

    except Exception as e:
        logging.error(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
