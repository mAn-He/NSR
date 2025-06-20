import argparse
import json
import sys
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import os

# Assuming models_final and engine_final are in paths accessible by Python
# For example, if they are in ./models/ and ./engine/ directories respectively:
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.models_final import MultiModalModel
from engine.engine_final import (
    set_random_seed,
    datapreprocessing,
    AdjustedSMAPELoss,
    # train_and_evaluate, # Using run_experiment instead
    # predict_and_evaluate,
    run_experiment,
    inv_boxcox_torch, # For direct use if needed, though often encapsulated
    # boxcox_standardscaling # The main scripts do custom boxcox, so direct use of this might vary
)

# From scipy, as used in original scripts for initial BoxCox lambda calculation
from scipy.stats import boxcox
# from scipy.special import inv_boxcox,boxcox1p,inv_boxcox1p # inv_boxcox_torch is in engine

# Placeholder for KNN specific function if not moved to engine_final.py
# from engines.week16_engines_knn_no_box import get_nearest_neighbors
def get_nearest_neighbors(*args, **kwargs):
    print("Warning: get_nearest_neighbors is a placeholder and needs to be implemented or correctly imported.")
    # This function should return df_train, df_test with KNN features
    # For now, it will just return the input dataframes to allow script structure to be built
    if 'df_train' in kwargs and 'df_test' in kwargs:
        return kwargs['df_train'], kwargs['df_test']
    elif len(args) > 1:
        return args[0], args[1]
    raise ValueError("df_train and df_test must be provided to get_nearest_neighbors placeholder.")


def none_or_int(value):
    if isinstance(value, str) and value.lower() == 'none':
        return None
    return int(value)

def main():
    parser = argparse.ArgumentParser(description='Unified Training Script')

    # Common arguments from all train scripts
    parser.add_argument('--device', type=int, default=0, help='Device number for GPU accelerating (default: 0)')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size (default: 64)')
    parser.add_argument('--image_embeddings_dim_out', type=none_or_int, default=128, help='Output dimension for image embeddings (default: 128, "None" for no linear proj)')
    parser.add_argument('--text_embeddings_dim_out', type=none_or_int, default=128, help='Output dimension for text embeddings (default: 128, "None" for no linear proj)')
    parser.add_argument('--other_features_dim_out', type=none_or_int, default=None, help='Output dimension for other features embeddings (default: None, "None" for no linear proj)')
    parser.add_argument('--header_mode', type=str, default='FFN', help='Header mode for the model (default: FFN)')
    parser.add_argument('--header_hidden_dims', type=int, nargs='+', default=[128], help='Hidden dimensions for the header (default: [128])')
    parser.add_argument('--dir_path', type=str, default='results/my_experiment', help='Directory path to save results (default: "results/my_experiment")')
    parser.add_argument('--train_dataset', type=str, default='../dataset/nsr_train_processed.csv', help='Train dataset path')
    parser.add_argument('--test_dataset', type=str, default='../dataset/nsr_test_processed.csv', help='Test dataset path')
    parser.add_argument('--random_seed', type=int, default=42, help='Random seed (default: 42)')
    parser.add_argument('--image_size', type=int, default=256, help='Image size for transforming (default: 256)')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate (default: 1e-4)') # Adjusted from very low 1e-7
    parser.add_argument('--num_epochs', type=int, default=50, help='Number of epochs for training (default: 50)')
    parser.add_argument('--loss_ratio', type=float, default=1.0, help='Scaled/Unscaled Loss ratio (default: 1.0, for scaled loss)')
    parser.add_argument('--image_normalizing', type=bool, default=False, help='Whether to use image normalization (default: False)')
    parser.add_argument('--nhead', type=int, default=4, help='Transformer nhead (default: 4)')
    parser.add_argument('--num_encoder_layers', type=int, default=6, help='Transformer num encoder layers (default: 6)')
    parser.add_argument('--num_decoder_layers', type=int, default=6, help='Transformer num decoder layers (default: 6)')
    parser.add_argument('--activation_func', type=str, default='gelu', help='Activation function (gelu or relu, default: gelu)')

    # KNN-specific arguments (from train_knn.py)
    parser.add_argument('--use_knn_features', action='store_true', help='Enable KNN feature processing')
    parser.add_argument('--knn_metric', type=str, default='euclidean', help='KNN distance metric (default: euclidean)')
    parser.add_argument('--k_neighbors', type=int, default=3, help='Number of K neighbors for KNN (default: 3)')
    parser.add_argument('--knn_embedding_metadata_concat', type=bool, default=False, help='Concatenate KNN embedding with metadata (default: False)')

    # Argument for choosing target scaling strategy
    parser.add_argument('--target_scaling', type=str, default='boxcox_custom', choices=['none', 'boxcox_custom'],
                        help='Target scaling strategy (default: boxcox_custom)')

    args = parser.parse_args()

    # Setup
    set_random_seed(args.random_seed)
    os.makedirs(args.dir_path, exist_ok=True)
    config_path = os.path.join(args.dir_path, 'config.json')
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(vars(args), f, indent=4, ensure_ascii=False)

    print(f"Configuration saved to {config_path}")
    print(f"Using device: cuda:{args.device}" if torch.cuda.is_available() else "Using device: cpu")
    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')

    # Load data
    df_train_orig = pd.read_csv(args.train_dataset, index_col=0 if 'train_knn' in args.train_dataset else None)
    df_test_orig = pd.read_csv(args.test_dataset, index_col=0 if 'train_knn' in args.test_dataset else None)

    # Drop unnecessary columns (common across original scripts)
    # Note: 'Unnamed: 0' might be an index column, handled by index_col in read_csv for knn variant
    cols_to_drop = ['Unnamed: 0', '판매시작연도', '판매첫날', '상품코드', '판매일자', '상품명', '상품명2',
                    '칼라', '칼라명', '칼라명2', '현재가', '할인율(%)', '파일경로',
                    '이미지갯수', '외관설명', '기능설명','카테고리']
    if not args.use_knn_features: # KNN script has different drop logic post get_nearest_neighbors
         df_train = df_train_orig.drop(columns=cols_to_drop, errors='ignore')
         df_test = df_test_orig.drop(columns=cols_to_drop, errors='ignore')
    else: # For KNN, less columns are dropped initially
        df_train = df_train_orig.copy() # KNN preprocessing will handle columns
        df_test = df_test_orig.copy()


    # Keep original dataframes for grouping in final evaluation (as done in week16_engines5.py)
    # These should have the '카테고리' column if it was originally present for grouping
    df_train_true_for_grouping = pd.read_csv(args.train_dataset, index_col=0 if 'train_knn' in args.train_dataset else None)
    df_test_true_for_grouping = pd.read_csv(args.test_dataset, index_col=0 if 'train_knn' in args.test_dataset else None)


    # Target Scaling (BoxCox with custom shift as in original scripts)
    a_param, b_param, lambda_train_boxcox = None, None, None
    is_scaled_target_used = False
    if args.target_scaling == 'boxcox_custom':
        y_train = np.array(df_train['판매수량'].tolist())
        # y_test = np.array(df_test['판매수량'].tolist()) # Not used for fitting lambda

        # Ensure positivity for boxcox
        if (y_train <= 0).any():
            print("Warning: y_train contains non-positive values. Adding 1 before Box-Cox.")
            y_train = y_train + 1
            # y_test = y_test + 1 # Apply consistently if needed for test set direct calc

        boxcox_y_train, lambda_train_boxcox = boxcox(y_train)

        # Custom scaling parameters 'a' and 'b'
        # Ensure quantiles are calculated on the original y_train (before +1 offset if applied for boxcox only)
        # This part needs careful check if y_train was modified for boxcox input only
        y_train_for_quantiles = np.array(df_train_orig['판매수량'].tolist())
        if (y_train_for_quantiles <=0).any() and (y_train == y_train_for_quantiles + 1).all():
             # if y_train was y_train_for_quantiles + 1, then (y_train_for_quantiles**lambda) might be problematic
             # The original scripts apply boxcox directly, implying y_train must be positive.
             # For safety, let's use the potentially shifted y_train for quantile calculation's base for power
             # if the original had non-positive values. This is complex.
             # Safest: use the y_train that went *into* boxcox for quantile calculation's base.
             pass # y_train is already the one that went into boxcox

        lowerbound = np.quantile(y_train, 0.16) # Use y_train that was input to boxcox
        upperbound = np.quantile(y_train, 0.84)

        # Handle lambda near zero for power calculations
        if abs(lambda_train_boxcox) < 1e-6: # lambda is zero
            term_lower = np.log(lowerbound)
            term_upper = np.log(upperbound)
        else: # lambda is not zero
            term_lower = np.power(lowerbound, lambda_train_boxcox)
            term_upper = np.power(upperbound, lambda_train_boxcox)

        a_param = 2 / (term_lower - term_upper)
        b_param = a_param * term_lower - 1

        df_train['판매수량_scaled'] = np.round(a_param * boxcox_y_train - b_param, 4)

        # Apply to test set using train lambda
        y_test_orig = np.array(df_test['판매수량'].tolist())
        if (y_test_orig <= 0).any(): y_test_orig = y_test_orig + 1 # Consistent shift if train was shifted
        boxcox_y_test = boxcox(y_test_orig, lmbda=lambda_train_boxcox)
        df_test['판매수량_scaled'] = np.round(a_param * boxcox_y_test - b_param, 4)

        is_scaled_target_used = True
        print(f"Applied custom BoxCox scaling. Lambda={lambda_train_boxcox:.4f}, a={a_param:.4f}, b={b_param:.4f}")
    else: # 'none'
        df_train['판매수량_scaled'] = df_train['판매수량'] # Or handle absence in dataset/engine
        df_test['판매수량_scaled'] = df_test['판매수량']
        print("No target scaling applied.")

    # KNN Feature Engineering (conditional)
    if args.use_knn_features:
        print("Applying KNN feature engineering...")
        # Ensure get_nearest_neighbors is correctly defined/imported and handles all necessary args
        df_train, df_test = get_nearest_neighbors(
            df_train=df_train, df_test=df_test, device=device, k=args.k_neighbors,
            dis_metric=args.knn_metric, image_normalizing=args.image_normalizing,
            image_resize_shape=(args.image_size, args.image_size),
            image_embedding_dim=args.image_embeddings_dim_out # This was image_embeddings_dim_out in train_knn
        )
        # train_knn.py drops more columns AFTER this step. Replicate that.
        cols_to_drop_knn = ['Unnamed: 0', '판매시작연도', '판매첫날', '상품코드', '판매일자', '상품명', '상품명2',
                            '칼라', '칼라명', '칼라명2', '현재가', '할인율(%)', '파일경로',
                            '이미지갯수', '외관설명', '기능설명','카테고리',
                            f'{args.k_neighbors} closest idx'] #This column is created by get_nearest_neighbors
        df_train = df_train.drop(columns=cols_to_drop_knn, errors='ignore')
        # Ensure test_df has same columns as train_df before dataloader
        df_test = df_test[df_train.columns.drop('판매수량_scaled',errors='ignore').tolist() + ['판매수량_scaled']]


    # Define embedding dimensions (after potential KNN processing)
    text_embedding_dim_in = len(df_train.filter(like='설명').columns)
    # Careful with column subtractions for other_features_dim_in
    # Base columns: all except '판매수량', '이미지파일', '판매수량_scaled', and text description columns
    potential_other_cols = df_train.drop(columns=df_train.filter(like='설명').columns.tolist() + \
                                           ['판매수량', '이미지파일', '판매수량_scaled'], errors='ignore')
    other_features_dim_in = potential_other_cols.shape[1]

    # If KNN features were added, they are part of other_features_dim_in
    # The original train_knn.py had: other_features_dim_in = df_train.drop(df_train.filter(like='설명').columns,axis=1).shape[-1]-4
    # This needs to be reconciled. Assuming KNN features are now part of the dataframe passed to dataloader.
    # The -3 or -4 in original scripts was to subtract target and other non-feature cols.
    # My calculation of other_features_dim_in above should be more robust.

    print(f"Text embedding input dimension: {text_embedding_dim_in}")
    print(f"Other features input dimension: {other_features_dim_in}")

    # Model Instantiation
    model_params = {
        'num_images_per_data': 2, # Assuming 2 from original scripts, make this an arg if it varies
        'header_mode': args.header_mode,
        'image_embeddings_dim_out': args.image_embeddings_dim_out,
        'text_embedding_dim_in': text_embedding_dim_in,
        'text_embedding_dim_out': args.text_embeddings_dim_out,
        'other_features_dim_in': other_features_dim_in,
        'other_features_dim_out': args.other_features_dim_out,
        'header_hidden_dims': args.header_hidden_dims,
        'nhead': args.nhead,
        'num_encoder_layers': args.num_encoder_layers,
        'num_decoder_layers': args.num_decoder_layers,
        'activation': args.activation_func
    }
    if args.use_knn_features:
        # Add KNN specific model params if the model architecture supports them
        # The MultiModalModel in models_final currently does not have explicit knn args
        # This implies either MultiModalModel needs updating, or KNN features are part of 'other_features'
        # train_knn.py used a different model (models_knn.py) which had knn_embedding_dim etc.
        # For now, assuming the consolidated MultiModalModel handles KNN features via other_features.
        # If specific architecture changes are needed for KNN, models_final.py would need that.
        print("Warning: KNN features are assumed to be part of 'other_features' for MultiModalModel.")
        # model_params['knn_embedding_dim'] = args.k_neighbors # Example if model took it
        # model_params['knn_embedding_metadata_concat'] = args.knn_embedding_metadata_concat

    model = MultiModalModel(**model_params)
    model.to(device)

    # Optimizer and Scheduler
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.1) # weight_decay from train_knn
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.2) # from train_knn

    # Loss Function and Evaluation Metric
    criterion = None
    # MSE-related: The following line instantiates MSELoss. It's being commented out as per requirements.
    # criterion = nn.MSELoss()
    print("# MSE-related: criterion = nn.MSELoss() # This line was commented out.")
    if criterion is None:
        print("Warning: Criterion is None because nn.MSELoss() was commented out. Training might fail or use a default.")
        # Fallback to a different loss or raise an error. For now, using AdjustedSMAPELoss also as criterion.
        # This is not ideal as SMAPE is not typically used for backprop directly in this form.
        # Consider nn.L1Loss() or requiring the user to specify.
        criterion = AdjustedSMAPELoss()
        print(f"Using {criterion.__class__.__name__} as a fallback criterion.")

    eval_metric = AdjustedSMAPELoss()

    # Prepare for engine call
    # The run_experiment function expects an args_config dict for dataloader & boxcox params
    engine_args_config = vars(args).copy()
    engine_args_config['a_param'] = a_param
    engine_args_config['b_param'] = b_param
    engine_args_config['boxcox_lambda'] = lambda_train_boxcox

    # Call the unified engine function
    run_experiment(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        eval_metric=eval_metric,
        device=device,
        train_df=df_train, # Processed df_train
        test_df=df_test,   # Processed df_test
        df_train_true_for_grouping=df_train_orig, # Original df for grouping in evaluation
        df_test_true_for_grouping=df_test_orig,   # Original df for grouping in evaluation
        num_epochs=args.num_epochs,
        dir_path=args.dir_path,
        args_config=engine_args_config,
        scheduler=scheduler,
        loss_ratio=args.loss_ratio,
        is_scaled_target=is_scaled_target_used, # True if boxcox_custom was used
        save_best_metric='eval_unscaled' # Example: or 'loss_unscaled', 'eval_scaled' etc.
    )

if __name__ == '__main__':
    main()
