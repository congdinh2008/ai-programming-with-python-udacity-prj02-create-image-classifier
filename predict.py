import argparse
import logging
from utils import load_checkpoint, set_device, predict, load_label_mapping

# Configure logging
logging.basicConfig(level=logging.INFO)


def main():
    arg_parser = argparse.ArgumentParser("predict.py")

    arg_parser.add_argument(
        "input",
        nargs="?",
        action="store",
        default="flowers/test/1/image_06752.jpg",
        type=str,
    )
    arg_parser.add_argument(
        "checkpoint", nargs="?", action="store", default="./checkpoint.pth", type=str
    )
    arg_parser.add_argument(
        "--top_k", dest="top_k", action="store", default=5, type=int
    )
    arg_parser.add_argument(
        "--category_names",
        dest="category_names",
        action="store",
        default="cat_to_name.json",
        type=str,
    )
    arg_parser.add_argument(
        "--gpu", dest="gpu", action="store", default="gpu", type=str
    )

    params = arg_parser.parse_args()

    # Load label mapping
    cat_to_name = load_label_mapping(params.category_names)
    if cat_to_name is None:
        return

    # Set device
    device = set_device(params.gpu)
    logging.info(f"Using device: {device}")

    # Load model
    try:
        model, _, _ = load_checkpoint(params.checkpoint, device)
        logging.info("Model loaded successfully.")
    except Exception as e:
        logging.error(f"Error loading model: {e}")
        return

    # Predict
    try:
        probs, classes, flowers = predict(params.input, model, device, cat_to_name, params.top_k)
        if probs is None or classes is None:
            raise ValueError("Prediction failed.")

        # Print prediction results
        class_names = [cat_to_name[i] for i in classes]
        for i in range(len(probs)):
            print(f"{class_names[i]} ({round(probs[i], 3) * 100}%)")

        logging.info("Prediction complete.")
    except Exception as e:
        logging.error(f"Error during prediction: {e}")


if __name__ == "__main__":
    main()
