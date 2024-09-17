# AI Programming with Python Project

Project code for Udacity's AI Programming with Python Nanodegree program. In this project, students first develop code for an image classifier built with PyTorch, then convert it into a command line application.

## Dependencies

The code is written in Python, and uses PyTorch. The dependencies are listed in the `requirements.txt` file. You can install them by running:

```bash
pip install -r requirements.txt
```

## Image Classifier

Project has 2 ipynb files:
1. Image Classifier Project.ipynb: This is the main project file.
2. Image Classifier Project with External Functions.ipynb: This file contains the same code as the main project file but with external functions.

The image classifier is built with PyTorch. The code is in the `Image Classifier Project.ipynb` Jupyter notebook. You can run the notebook by typing:

```bash
jupyter notebook Image\ Classifier\ Project.ipynb
```

## Command Line Application

The command line application is in the `download_data.py`, `train.py` and `predict.py` files. You can train the network by running:

```bash
python train.py
```

And you can make predictions by running:

```bash
python predict.py
```

The `predict.py` script has the following arguments:

- Required:
  - `input`: The path to an image file.
  - `checkpoint`: The path to the checkpoint file.
- Optional:
    - `--top_k`: Return the top KK most likely classes.
    - `--category_names`: A mapping of categories to real names.
    - `--gpu`: Use GPU for inference.

For example, you can run:
    
    ```bash
    python predict.py input checkpoint --top_k 3 --category_names cat_to_name.json --gpu
    ```