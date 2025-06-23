# Multimodal Sales Prediction Model

This project implements a multimodal deep learning model for sales prediction. It leverages various data sources including product images, textual descriptions (product names, functional and appearance descriptions), and other tabular features. The pipeline includes comprehensive data preprocessing, feature engineering (e.g., RoBERTa embeddings for text, ResNet features for images, KNN-based features), and a flexible model architecture to combine these diverse inputs for accurate sales forecasting.

## Project Structure

The project is organized into the following main directories:

-   `/models`: Contains the definitions for the neural network architectures.
    -   `models_final.py`: The consolidated script defining the primary multimodal model used in the project.
-   `/engine`: Includes the core logic for the training and evaluation pipeline.
    -   `engine_final.py`: The consolidated script that manages training epochs, evaluation, model saving, and logging.
-   `/main`: Houses the main executable scripts for running experiments.
    -   `main_final.py`: The primary script to configure and initiate the model training and evaluation process.
-   `/prework`: Contains Jupyter notebooks used for data preprocessing, exploratory data analysis (EDA), and complex feature engineering.
    -   `prework_final.ipynb`: A consolidated notebook detailing steps for data loading, cleaning, text processing, image path handling, target variable transformation (BoxCox), and KNN-based feature generation.
-   `/dataset`: (Assumed) Should contain the necessary dataset files. The scripts may reference paths like `../dataset/your_train_data.csv`.

## Features

-   **Multimodal Data Integration:** Combines image data, textual descriptions, and other tabular features for sales prediction.
-   **Deep Learning Model:** Utilizes a flexible neural network architecture (configurable header with FFN, Transformer options) built with PyTorch.
-   **Image Embeddings:** Employs ResNet (e.g., ResNet152) to generate feature embeddings from product images.
-   **Text Embeddings:** Uses KLUE/RoBERTa-base model to create contextual embeddings from Korean product descriptions (original data language).
-   **Feature Engineering:**
    -   **BoxCox Transformation:** Applied to the target variable (`sales_quantity`) to handle skewness and stabilize variance.
    -   **KNN-based Features:** Optionally generates features based on K-Nearest Neighbors in the combined feature space, considering neighbor sales data.
-   **Comprehensive Preprocessing:** Includes detailed steps for data cleaning, color mapping, text processing, and image path handling, as documented in `prework/prework_final.ipynb`.
-   **Structured Training Pipeline:** The `engine/engine_final.py` script provides a robust training and evaluation loop with support for:
    -   Checkpointing (saving best models).
    -   Learning rate scheduling.
    -   TensorBoard logging (if configured).
    -   Customizable loss functions (AdjustedSMAPELoss used, MSE commented out).
-   **Code Organization:** Scripts are organized into `main`, `models`, `engine`, and `prework` directories for clarity. Consolidated `_final` files represent the primary workflows.

## Setup

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```

2.  **Create a Python virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3.  **Install dependencies:**
    While a `requirements.txt` is not explicitly provided, you will need common data science and deep learning libraries. Key libraries used include:
    -   `pandas`
    -   `numpy`
    -   `torch` (PyTorch)
    -   `torchvision`
    -   `scikit-learn`
    -   `scipy`
    -   `transformers` (for RoBERTa embeddings)
    -   `matplotlib`
    -   `seaborn`
    -   `tqdm`

    You can typically install these using pip:
    ```bash
    pip install pandas numpy torch torchvision scikit-learn scipy transformers matplotlib seaborn tqdm
    ```
    Ensure you install a version of PyTorch compatible with your CUDA setup if you plan to use a GPU.

4.  **Dataset:**
    -   Prepare your dataset and place it in a directory (e.g., `dataset/`).
    -   The main training script (`main/main_final.py`) expects paths to train and test CSV files to be provided as command-line arguments (e.g., `--train_dataset ../dataset/your_train_data.csv`).
    -   The dataset should contain columns as expected by the preprocessing steps outlined in `prework/prework_final.ipynb` and used by `main/main_final.py`.

## How to Run

The main training and evaluation script is `main/main_final.py`.

1.  **Navigate to the project root directory.**

2.  **Execute the script using `python -m main.main_final`:**
    This ensures that relative imports within the project are handled correctly.

    **Basic Example Command:**
    ```bash
    python -m main.main_final \
        --train_dataset dataset/your_train_data.csv \
        --test_dataset dataset/your_test_data.csv \
        --dir_path results/experiment_01
    ```
    *Note: Adjust `dataset/your_train_data.csv` and `dataset/your_test_data.csv` to the actual paths of your data files relative to the project root.*

3.  **Command-Line Arguments:**
    The script accepts various arguments to configure the experiment. Some common ones include:
    -   `--train_dataset <path>`: Path to your training CSV file.
    -   `--test_dataset <path>`: Path to your testing CSV file.
    -   `--dir_path <path>`: Directory to save outputs (configs, models, logs).
    -   `--num_epochs <number>`: Number of training epochs.
    -   `--batch_size <number>`: Batch size.
    -   `--learning_rate <float>`: Learning rate.
    -   `--device <0, 1, ...>`: GPU device ID.
    -   `--use_knn_features`: Add this flag to enable KNN feature processing.

    For a full list of arguments and their default values, run:
    ```bash
    python -m main.main_final --help
    ```

## A Note on Column Naming

The column names used throughout the codebase (Python scripts and Jupyter notebooks) and expected in the input data CSVs are in English. These have been translated from original Korean column names for broader accessibility and standardization.
