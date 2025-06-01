import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch.nn as nn
import torchvision
import pandas as pd
from torchvision import transforms
from PIL import Image
import numpy as np
import ast
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg') # Ensure backend is set before pyplot import for headless environments
import matplotlib.pyplot as plt
import os
import random
from scipy.stats import boxcox
from scipy.special import inv_boxcox, boxcox1p, inv_boxcox1p
from torch.utils.tensorboard import SummaryWriter # Added from week16_engines5.py

def set_random_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class AdjustedSMAPELoss(nn.Module):
    def __init__(self, epsilon=1e-6):
        super(AdjustedSMAPELoss, self).__init__()
        self.epsilon = epsilon

    def forward(self, y_pred, y_true):
        numerator = torch.abs(y_pred - y_true)
        denominator = torch.abs(y_pred) + torch.abs(y_true) + self.epsilon
        smape_score = torch.mean(numerator / denominator * 2) / 2 # Original had /2, keeping it
        return smape_score

def calculate_ad_smape_grouped_by(df_pred, df_true, eval_metric):
    # Standardizing column names for merge
    if '칼라' in df_pred.columns and '칼라명2' in df_true.columns:
        df_true_renamed = df_true.rename(columns={'칼라명2': '칼라'})
        df = df_pred.merge(df_true_renamed[['상품코드', '칼라', '카테고리']], how='left', on=['상품코드', '칼라'])
    elif '상품코드' in df_pred.columns and '상품코드' in df_true.columns and '카테고리' in df_true.columns:
        df = df_pred.merge(df_true[['상품코드', '카테고리']], how='left', on='상품코드')
    else:
        # Fallback or error if key columns for grouping are missing
        print("Warning: Key columns for grouping in calculate_ad_smape_grouped_by are missing.")
        return {}

    cats = df['카테고리'].unique()
    dictionary = {}

    for cat in cats:
        if pd.isna(cat): # Handle potential NaN categories
            continue
        x = df[df['카테고리'] == cat]
        y1 = torch.tensor(x['판매수량_true'].values, dtype=torch.float32)
        y2 = torch.tensor(x['판매수량_pred'].values, dtype=torch.float32)

        eval_score = eval_metric(y1, y2)
        dictionary[cat] = eval_score.item() if hasattr(eval_score, 'item') else eval_score

    return dictionary

def inv_boxcox_torch(x, a, b, lmbda):
    epsilon = 1e-6
    if abs(lmbda) < epsilon: # Check if lambda is close to 0
        return torch.exp((x + b) / a) # Corrected from original to match typical inv_boxcox for lambda=0
    else:
        # Ensure base is non-negative for fractional powers if lmbda is not an integer
        # This might require ((x + b) / a).clamp(min=epsilon) depending on typical x, a, b values
        return torch.pow(((x + b) / a), 1 / lmbda)

# Placeholder for nsr_img_txt_dataset, datapreprocessing, train, test, run functions
# These will be added in subsequent steps.

class nsr_img_txt_dataset(Dataset):
    def __init__(self, dataframe, transform=None, fixed_num_images=2):
        self.dataframe = dataframe
        self.transform = transform
        self.fixed_num_images = fixed_num_images

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        image_paths_str = row['이미지파일']

        # Robustly evaluate image_paths_str
        try:
            image_paths = ast.literal_eval(image_paths_str)
            if not isinstance(image_paths, list):
                # Handle cases where literal_eval doesn't return a list (e.g. if it's a single string path)
                image_paths = [str(image_paths)]
        except (ValueError, SyntaxError):
            # If ast.literal_eval fails, assume it's a single path or needs specific handling
            # This part might need adjustment based on actual data format variations
            image_paths = [str(image_paths_str)]


        text_embeddings = row.filter(like='설명').values.astype(np.float32)
        target = row['판매수량']
        target_scaled = row.get('판매수량_scaled', None) # Use .get for optional column

        other_embeddings_cols = row.drop(labels=row.filter(like='설명').index.tolist() + ['판매수량','이미지파일','판매수량_scaled'], errors='ignore')
        other_embeddings = other_embeddings_cols.values.astype(np.float32)

        images = []
        valid_image_paths = [p for p in image_paths if isinstance(p, str) and os.path.exists(p)]

        for image_path in valid_image_paths:
            try:
                img = Image.open(image_path).convert('RGB') # Ensure RGB
                if self.transform:
                    img = self.transform(img)
                images.append(img)
            except Exception as e:
                print(f"Warning: Could not load image {image_path}. Error: {e}")
                # Optionally, append a placeholder image if an image fails to load
                # images.append(torch.zeros((3, self.transform.transforms[0].size[0], self.transform.transforms[0].size[1])) if self.transform else torch.zeros((3, 256, 256)))


        if not images: # If no valid images were loaded
            print(f"Warning: No valid images found for row {idx}. Appending zeros.")
            # Determine size from transform if possible, else default
            size = (256, 256)
            if self.transform:
                for t in self.transform.transforms:
                    if isinstance(t, transforms.Resize):
                        size = t.size if isinstance(t.size, tuple) else (t.size, t.size)
                        break
            placeholder_image = torch.zeros((3, size[0], size[1]))
            for _ in range(self.fixed_num_images):
                images.append(placeholder_image)

        # Ensure fixed number of images
        while len(images) < self.fixed_num_images and images: # if images is not empty
            images.append(images[-1]) # Duplicate last valid image

        if len(images) > self.fixed_num_images:
            images = images[:self.fixed_num_images]

        images_tensor = torch.stack(images)

        text_embeddings_tensor = torch.tensor(text_embeddings, dtype=torch.float32)
        other_embeddings_tensor = torch.tensor(other_embeddings, dtype=torch.float32)
        target_tensor = torch.tensor(target, dtype=torch.float32)

        if target_scaled is not None:
            target_scaled_tensor = torch.tensor(target_scaled, dtype=torch.float32)
            return images_tensor, text_embeddings_tensor, other_embeddings_tensor, target_tensor, target_scaled_tensor
        else:
            return images_tensor, text_embeddings_tensor, other_embeddings_tensor, target_tensor

def datapreprocessing(train_df, test_df, dataloader_generator, image_resize_shape=(256, 256), image_normalizing=False, batch_size=64, shuffle_train=True, shuffle_test=False, num_workers=None, pin_memory=False):
    """
    train_df : DataFrame for Training
    test_df : DataFrame for Testing
    dataloader_generator : PyTorch Generator for reproducible DataLoader shuffling
    image_resize_shape : image_size to reshape
    image_normalizing : whether to apply normalization
    batch_size : Batch size
    shuffle_train : whether to shuffle training data
    shuffle_test : whether to shuffle test data (usually False)
    """

    if image_normalizing:
        transform = transforms.Compose([
            transforms.Resize(image_resize_shape),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize(image_resize_shape),
            transforms.ToTensor()
        ])

    print("Applied image transform:")
    print(transform)

    train_dataset = nsr_img_txt_dataset(train_df, transform=transform)
    test_dataset = nsr_img_txt_dataset(test_df, transform=transform)

    effective_num_workers = num_workers if num_workers is not None else os.cpu_count()
    print(f"Using {effective_num_workers} workers for DataLoaders.")

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train,
        generator=dataloader_generator,
        num_workers=effective_num_workers,
        pin_memory=pin_memory
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=shuffle_test, # Test data is typically not shuffled
        generator=dataloader_generator, # Still useful for reproducibility if shuffle_test were True
        num_workers=effective_num_workers,
        pin_memory=pin_memory
    )

    return train_dataloader, test_dataloader

# Placeholder for train, test, run functions
# These will be added in subsequent steps.

def boxcox_standardscaling(df_column_series, lmbda=None):
    """
    Applies Box-Cox transformation and then StandardScaler to a pandas Series.
    Args:
        df_column_series (pd.Series): The series to transform.
        lmbda (float, optional): Lambda for Box-Cox. If None, it's found automatically.
    Returns:
        transformed_scaled_series (pd.Series): Transformed and scaled series.
        optimal_lambda (float): The lambda used for Box-Cox.
        scaler (StandardScaler): The fitted scaler.
    """
    from sklearn.preprocessing import StandardScaler

    # Ensure series is positive for Box-Cox
    if (df_column_series <= 0).any():
        print("Warning: Data contains non-positive values. Adding 1 before Box-Cox transformation.")
        # Adding a small constant, ensure this is appropriate for your data context
        series_transformed, optimal_lambda = boxcox(df_column_series + 1, lmbda=lmbda)
    else:
        series_transformed, optimal_lambda = boxcox(df_column_series, lmbda=lmbda)

    scaler = StandardScaler()
    scaled_values = scaler.fit_transform(series_transformed.reshape(-1, 1))
    return scaled_values.flatten(), optimal_lambda, scaler

def inv_scaling_boxcox(scaled_values, optimal_lambda, scaler, a_shift=None, b_shift=None):
    """
    Reverses StandardScaler and then Box-Cox transformation.
    Includes optional a and b parameters for the custom scaling: scaled_y = a * (y**lambda) - b
    If a_shift and b_shift are provided, it assumes the inv_boxcox_torch logic.
    Otherwise, uses standard scipy inv_boxcox.
    Args:
        scaled_values (np.array): Values to inverse transform.
        optimal_lambda (float): Lambda used for Box-Cox.
        scaler (StandardScaler): Scaler used for scaling.
        a_shift (float, optional): 'a' parameter from custom scaling.
        b_shift (float, optional): 'b' parameter from custom scaling.
    Returns:
        original_values (np.array): The inverse transformed values.
    """
    unscaled_values = scaler.inverse_transform(scaled_values.reshape(-1, 1)).flatten()
    if a_shift is not None and b_shift is not None:
        # Use the torch-compatible inverse Box-Cox for the custom scaling
        # Ensure input is a tensor for inv_boxcox_torch
        unscaled_tensor = torch.tensor(unscaled_values, dtype=torch.float32)
        original_values_tensor = inv_boxcox_torch(unscaled_tensor, a_shift, b_shift, optimal_lambda)
        original_values = original_values_tensor.cpu().numpy()
    else:
        # Standard scipy inverse Box-Cox
        original_values = inv_boxcox(unscaled_values, optimal_lambda)
        # If 1 was added during transformation for non-positive values
        # This needs to be tracked and subtracted here if that was the case globally.
        # For now, assuming direct inverse. Consider passing a flag if offset was added.

    return original_values

# More robust training and evaluation structure
def train_one_epoch(model, dataloader, criterion, optimizer, device,
                    loss_ratio=1.0, eval_metric=None,
                    a_param=None, b_param=None, boxcox_lambda=None,
                    is_scaled_target=True):
    model.train()
    total_loss = 0.0
    total_eval_metric_scaled = 0.0 # For scaled targets if applicable
    total_eval_metric_unscaled = 0.0 # For unscaled targets

    # Freeze ResNet body, unfreeze FC head if it's part of the model structure
    # This was common in the original scripts. Adapt if model structure varies.
    if hasattr(model, 'image_model') and hasattr(model.image_model, 'fc'):
        for param in model.image_model.parameters():
            param.requires_grad = False
        for param in model.image_model.fc.parameters():
            param.requires_grad = True

    for batch in tqdm(dataloader, desc='Training Epoch', leave=False):
        images, text_embeddings, other_embeddings, targets_unscaled, targets_scaled = [x.to(device) for x in batch]

        optimizer.zero_grad()

        outputs = model(images, text_embeddings, other_embeddings) # Model output is assumed to be 'scaled' if transformations are used

        current_loss = 0

        if is_scaled_target and targets_scaled is not None:
            # MSE-related: criterion(outputs.flatten(),targets_scaled)
            loss_scaled = criterion(outputs.flatten(), targets_scaled.flatten())
            if eval_metric:
                # MSE-related: eval_metric(outputs.flatten(), targets_scaled.flatten()) (if eval_metric is MSE)
                eval_m_scaled = eval_metric(outputs.flatten(), targets_scaled.flatten())
                total_eval_metric_scaled += eval_m_scaled.item()

            # Inverse transform output to unscaled for second part of loss/eval if needed
            if a_param is not None and b_param is not None and boxcox_lambda is not None:
                outputs_unscaled = inv_boxcox_torch(outputs.flatten(), a_param, b_param, boxcox_lambda)
                # MSE-related: criterion(outputs_unscaled, targets_unscaled.flatten())
                loss_unscaled = criterion(outputs_unscaled, targets_unscaled.flatten())
                if eval_metric:
                    # MSE-related: eval_metric(outputs_unscaled, targets_unscaled.flatten()) (if eval_metric is MSE)
                    eval_m_unscaled = eval_metric(outputs_unscaled, targets_unscaled.flatten())
                    total_eval_metric_unscaled += eval_m_unscaled.item()
                # MSE-related: loss_ratio * loss_scaled + (1-loss_ratio) * loss_unscaled
                current_loss = loss_ratio * loss_scaled + (1 - loss_ratio) * loss_unscaled
            else: # Only scaled loss contributes
                current_loss = loss_scaled
        else: # Working with unscaled targets directly
            # MSE-related: criterion(outputs.flatten(), targets_unscaled.flatten())
            current_loss = criterion(outputs.flatten(), targets_unscaled.flatten())
            if eval_metric:
                # MSE-related: eval_metric(outputs.flatten(), targets_unscaled.flatten()) (if eval_metric is MSE)
                eval_m_unscaled = eval_metric(outputs.flatten(), targets_unscaled.flatten())
                total_eval_metric_unscaled += eval_m_unscaled.item()

        if torch.isnan(current_loss):
            print(f"NaN detected in loss. Skipping batch.")
            continue

        current_loss.backward()
        optimizer.step()

        total_loss += current_loss.item()

    avg_loss = total_loss / len(dataloader)
    avg_eval_metric_scaled = total_eval_metric_scaled / len(dataloader) if total_eval_metric_scaled > 0 else 0
    avg_eval_metric_unscaled = total_eval_metric_unscaled / len(dataloader) if total_eval_metric_unscaled > 0 else 0

    return avg_loss, avg_eval_metric_scaled, avg_eval_metric_unscaled


def evaluate_one_epoch(model, dataloader, criterion, device,
                       loss_ratio=1.0, eval_metric=None,
                       a_param=None, b_param=None, boxcox_lambda=None,
                       is_scaled_target=True):
    model.eval()
    total_loss = 0.0
    total_eval_metric_scaled = 0.0
    total_eval_metric_unscaled = 0.0

    with torch.inference_mode():
        for batch in tqdm(dataloader, desc='Evaluating Epoch', leave=False):
            images, text_embeddings, other_embeddings, targets_unscaled, targets_scaled = [x.to(device) for x in batch]

            outputs = model(images, text_embeddings, other_embeddings)
            current_loss = 0

            if is_scaled_target and targets_scaled is not None:
                # MSE-related: criterion(outputs.flatten(),targets_scaled)
                loss_scaled = criterion(outputs.flatten(), targets_scaled.flatten())
                if eval_metric:
                    # MSE-related: eval_metric(outputs.flatten(), targets_scaled.flatten()) (if eval_metric is MSE)
                    eval_m_scaled = eval_metric(outputs.flatten(), targets_scaled.flatten())
                    total_eval_metric_scaled += eval_m_scaled.item()

                if a_param is not None and b_param is not None and boxcox_lambda is not None:
                    outputs_unscaled = inv_boxcox_torch(outputs.flatten(), a_param, b_param, boxcox_lambda)
                    # MSE-related: criterion(outputs_unscaled, targets_unscaled.flatten())
                    loss_unscaled = criterion(outputs_unscaled, targets_unscaled.flatten())
                    if eval_metric:
                        # MSE-related: eval_metric(outputs_unscaled, targets_unscaled.flatten()) (if eval_metric is MSE)
                        eval_m_unscaled = eval_metric(outputs_unscaled, targets_unscaled.flatten())
                        total_eval_metric_unscaled += eval_m_unscaled.item()
                    # MSE-related: loss_ratio * loss_scaled + (1-loss_ratio) * loss_unscaled
                    current_loss = loss_ratio * loss_scaled + (1 - loss_ratio) * loss_unscaled
                else:
                    current_loss = loss_scaled
            else: # Working with unscaled targets directly
                # MSE-related: criterion(outputs.flatten(), targets_unscaled.flatten())
                current_loss = criterion(outputs.flatten(), targets_unscaled.flatten())
                if eval_metric:
                     # MSE-related: eval_metric(outputs.flatten(), targets_unscaled.flatten()) (if eval_metric is MSE)
                    eval_m_unscaled = eval_metric(outputs.flatten(), targets_unscaled.flatten())
                    total_eval_metric_unscaled += eval_m_unscaled.item()

            total_loss += current_loss.item()

    avg_loss = total_loss / len(dataloader)
    avg_eval_metric_scaled = total_eval_metric_scaled / len(dataloader) if total_eval_metric_scaled > 0 else 0
    avg_eval_metric_unscaled = total_eval_metric_unscaled / len(dataloader) if total_eval_metric_unscaled > 0 else 0

    return avg_loss, avg_eval_metric_scaled, avg_eval_metric_unscaled


def train_and_evaluate(model, train_dataloader, test_dataloader, optimizer, criterion, eval_metric,
                       device, num_epochs, dir_path, scheduler=None, writer=None,
                       loss_ratio=1.0, a_param=None, b_param=None, boxcox_lambda=None,
                       is_scaled_target=True, save_best_metric='loss_unscaled'):

    os.makedirs(dir_path, exist_ok=True)

    history = {
        'train_loss': [], 'test_loss': [],
        'train_eval_scaled': [], 'test_eval_scaled': [],
        'train_eval_unscaled': [], 'test_eval_unscaled': []
    }

    best_metric_val = float('inf') if 'loss' in save_best_metric else float('-inf')
    best_model_state = None

    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")

        train_loss, train_eval_s, train_eval_u = train_one_epoch(
            model, train_dataloader, criterion, optimizer, device,
            loss_ratio, eval_metric, a_param, b_param, boxcox_lambda, is_scaled_target
        )

        test_loss, test_eval_s, test_eval_u = evaluate_one_epoch(
            model, test_dataloader, criterion, device,
            loss_ratio, eval_metric, a_param, b_param, boxcox_lambda, is_scaled_target
        )

        history['train_loss'].append(train_loss)
        history['test_loss'].append(test_loss)
        history['train_eval_scaled'].append(train_eval_s)
        history['test_eval_scaled'].append(test_eval_s)
        history['train_eval_unscaled'].append(train_eval_u)
        history['test_eval_unscaled'].append(test_eval_u)

        print(f"Train Results: Loss={train_loss:.4f}, EvalScaled={train_eval_s:.4f}, EvalUnscaled={train_eval_u:.4f}")
        print(f"Test Results: Loss={test_loss:.4f}, EvalScaled={test_eval_s:.4f}, EvalUnscaled={test_eval_u:.4f}")

        if writer:
            writer.add_scalar('Loss/train_epoch', train_loss, epoch + 1)
            writer.add_scalar('Loss/test_epoch', test_loss, epoch + 1)
            if train_eval_s > 0 : writer.add_scalar('EvalScaled/train_epoch', train_eval_s, epoch + 1)
            if test_eval_s > 0 : writer.add_scalar('EvalScaled/test_epoch', test_eval_s, epoch + 1)
            if train_eval_u > 0 : writer.add_scalar('EvalUnscaled/train_epoch', train_eval_u, epoch + 1)
            if test_eval_u > 0 : writer.add_scalar('EvalUnscaled/test_epoch', test_eval_u, epoch + 1)

        current_metric_val = 0
        if save_best_metric == 'loss_scaled': current_metric_val = test_loss # if primary loss is scaled
        elif save_best_metric == 'loss_unscaled': current_metric_val = test_eval_u if is_scaled_target and test_eval_u > 0 else test_loss
        elif save_best_metric == 'eval_scaled': current_metric_val = test_eval_s
        elif save_best_metric == 'eval_unscaled': current_metric_val = test_eval_u
        else: current_metric_val = test_loss # Default to test_loss

        if ('loss' in save_best_metric and current_metric_val < best_metric_val) or \
           ('eval' in save_best_metric and current_metric_val > best_metric_val):
            best_metric_val = current_metric_val
            best_model_state = model.state_dict()
            torch.save(best_model_state, os.path.join(dir_path, 'best_model.pt'))
            print(f"Saved best model at epoch {epoch + 1} with {save_best_metric}: {best_metric_val:.4f}")

        if scheduler:
            scheduler.step()

    # Plotting (simplified, can be expanded)
    # MSE-related: Consider commenting these plots if they are primarily for MSE loss/eval
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['test_loss'], label='Test Loss')
    plt.title('Total Loss') # MSE-related: if criterion is MSE
    plt.legend()

    if any(history['train_eval_scaled']) or any(history['test_eval_scaled']):
        plt.subplot(2, 2, 2)
        plt.plot(history['train_eval_scaled'], label='Train Eval Scaled')
        plt.plot(history['test_eval_scaled'], label='Test Eval Scaled')
        plt.title('Scaled Evaluation Metric') # MSE-related: if eval_metric is MSE
        plt.legend()

    if any(history['train_eval_unscaled']) or any(history['test_eval_unscaled']):
        plt.subplot(2, 2, 3)
        plt.plot(history['train_eval_unscaled'], label='Train Eval Unscaled')
        plt.plot(history['test_eval_unscaled'], label='Test Eval Unscaled')
        plt.title('Unscaled Evaluation Metric') # MSE-related: if eval_metric is MSE
        plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(dir_path, 'training_plots.png'))
    plt.close()

    print("Training complete. Plots saved.")
    if best_model_state:
        model.load_state_dict(best_model_state) # Load best model for returning
    return model, history


def predict_and_evaluate(model, dataloader, device, df_original, df_true_for_grouping,
                         criterion, eval_metric, output_dir, dataset_name,
                         a_param=None, b_param=None, boxcox_lambda=None, is_scaled_output=True):
    model.eval()
    predictions_list = []
    total_loss = 0.0
    total_eval = 0.0

    with torch.inference_mode():
        for batch in tqdm(dataloader, desc=f'Predicting on {dataset_name}', leave=False):
            images, text_embeddings, other_embeddings, targets_unscaled, targets_scaled = [x.to(device) for x in batch]

            outputs_model = model(images, text_embeddings, other_embeddings) # Assumed scaled if transformations were applied

            if is_scaled_output and a_param is not None and b_param is not None and boxcox_lambda is not None:
                outputs_final = inv_boxcox_torch(outputs_model.flatten(), a_param, b_param, boxcox_lambda)
            else:
                outputs_final = outputs_model.flatten()

            predictions_list.extend(outputs_final.cpu().numpy())

            # Calculate loss and eval against unscaled targets
            # MSE-related: criterion(outputs_final, targets_unscaled.flatten())
            loss = criterion(outputs_final, targets_unscaled.flatten())
            # MSE-related: eval_metric(outputs_final, targets_unscaled.flatten())
            eval_score = eval_metric(outputs_final, targets_unscaled.flatten())

            total_loss += loss.item()
            total_eval += eval_score.item()

    avg_loss = total_loss / len(dataloader)
    avg_eval = total_eval / len(dataloader)
    print(f"{dataset_name} Prediction: Avg Loss={avg_loss:.4f}, Avg Eval Metric={avg_eval:.4f}")

    df_preds = df_original.copy()
    # Ensure alignment if df_preds was from a shuffled dataloader (not typical for test)
    # This assumes df_preds corresponds row-wise to the dataloader's iteration order.
    # If dataloader was shuffled, original indices would be needed.
    df_preds['판매수량_pred'] = np.round(predictions_list[:len(df_preds)], 2) # Truncate if preds are more than df rows
    df_preds['판매수량_true'] = df_original['판매수량'] # Ensure true values are present for comparison

    pred_csv_path = os.path.join(output_dir, f"{dataset_name}_predictions.csv")
    df_preds.to_csv(pred_csv_path, encoding='utf-8', index=False)
    print(f"Predictions saved to {pred_csv_path}")

    # Grouped evaluation (if applicable)
    if df_true_for_grouping is not None:
        # For item_code level (상품코드별)
        df_item_agg = pd.DataFrame()
        df_item_agg['상품코드'] = df_preds['상품코드'].unique()
        df_item_agg = df_item_agg.merge(df_preds.groupby('상품코드')['판매수량_true'].sum().reset_index(), on='상품코드', how='left')
        df_item_agg = df_item_agg.merge(df_preds.groupby('상품코드')['판매수량_pred'].sum().reset_index(), on='상품코드', how='left')
        df_item_agg.to_csv(os.path.join(output_dir, f"{dataset_name}_item_level_preds.csv"), encoding='utf-8', index=False)

        item_level_eval = eval_metric(torch.tensor(df_item_agg['판매수량_pred'].values, dtype=torch.float32),
                                      torch.tensor(df_item_agg['판매수량_true'].values, dtype=torch.float32)).item()
        print(f"{dataset_name} Item-Level Eval Metric: {item_level_eval:.4f}")

        # Category-wise for item_code level
        cat_item_eval = calculate_ad_smape_grouped_by(df_item_agg, df_true_for_grouping, eval_metric)
        print(f"{dataset_name} Category-wise Item-Level Eval: {cat_item_eval}")

        # Category-wise for item_code + color level (original df_preds)
        cat_item_color_eval = calculate_ad_smape_grouped_by(df_preds, df_true_for_grouping, eval_metric)
        print(f"{dataset_name} Category-wise Item-Color-Level Eval: {cat_item_color_eval}")

    return avg_loss, avg_eval


def run_experiment(model, optimizer, criterion, eval_metric, device,
                   train_df, test_df, df_train_true_for_grouping, df_test_true_for_grouping, # df_true are original non-shuffled dfs for grouping
                   num_epochs, dir_path, args_config, # args_config for boxcox and dataloader params
                   scheduler=None, loss_ratio=1.0,
                   is_scaled_target=True, save_best_metric='loss_unscaled'):

    writer = SummaryWriter(log_dir=os.path.join(dir_path, 'tensorboard_logs'))

    # Data preprocessing
    # BoxCox parameters (a,b,lambda) might be determined from training data if applicable
    # For now, assuming they are part of args_config if used, or determined globally
    a_param, b_param, boxcox_lambda = args_config.get('a_param'), args_config.get('b_param'), args_config.get('boxcox_lambda')

    dataloader_generator = torch.Generator().manual_seed(args_config.get('random_seed', 42))
    train_dataloader, test_dataloader = datapreprocessing(
        train_df, test_df, dataloader_generator,
        image_resize_shape=(args_config.get('image_size', 256), args_config.get('image_size', 256)),
        image_normalizing=args_config.get('image_normalizing', False),
        batch_size=args_config.get('batch_size', 64),
        shuffle_train=True, shuffle_test=False, # Standard practice
        num_workers=args_config.get('num_workers', os.cpu_count()),
        pin_memory=args_config.get('pin_memory', True)
    )

    best_model, history = train_and_evaluate(
        model, train_dataloader, test_dataloader, optimizer, criterion, eval_metric,
        device, num_epochs, dir_path, scheduler, writer,
        loss_ratio, a_param, b_param, boxcox_lambda, is_scaled_target, save_best_metric
    )

    print("\n--- Final Evaluation on Training Set ---")
    predict_and_evaluate(
        best_model, train_dataloader, device, train_df, df_train_true_for_grouping, # train_df here is the one used for dataloader
        criterion, eval_metric, dir_path, "train_final",
        a_param, b_param, boxcox_lambda, is_scaled_target
    )

    print("\n--- Final Evaluation on Test Set ---")
    predict_and_evaluate(
        best_model, test_dataloader, device, test_df, df_test_true_for_grouping, # test_df here is the one used for dataloader
        criterion, eval_metric, dir_path, "test_final",
        a_param, b_param, boxcox_lambda, is_scaled_target
    )

    if writer:
        writer.close()

    print(f"Experiment finished. Results saved in {dir_path}")
    return best_model, history
