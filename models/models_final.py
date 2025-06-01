import argparse
import json
import sys
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import seaborn as sns
import os
from glob import glob
import matplotlib
matplotlib.use('Agg') # Ensure backend is set before pyplot import for headless environments
import matplotlib.pyplot as plt
from scipy.stats import boxcox
from scipy.special import inv_boxcox, boxcox1p, inv_boxcox1p
# from sklearn.metrics.pairwise import euclidean_distances, cosine_similarity # Only in knn_trial, maybe not for general model

# Helper classes ( взято из models_transformer_v3.py / knn_trial.py как наиболее полные)

class MLPRegression(nn.Module):
    def __init__(self, input_dim, hidden_dims, dropout_prob=0.2, activation='gelu'):
        super(MLPRegression, self).__init__()
        layers = []
        in_features = input_dim

        if activation.lower() == 'relu':
            self.activ = nn.ReLU()
        elif activation.lower() == 'gelu':
            self.activ = nn.GELU()
        else:
            # Default to GELU if activation is not recognized or add error handling
            self.activ = nn.GELU()
            print(f"Warning: MLPRegression activation '{activation}' not recognized. Defaulting to GELU.")

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_features, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))  # Batch Normalization
            layers.append(self.activ)
            layers.append(nn.Dropout(dropout_prob))  # Dropout
            in_features = hidden_dim

        layers.append(nn.Linear(in_features, 1))  # Output layer for regression

        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)

class Dense_block(nn.Module):
    def __init__(self, input_dim, hidden_dims, activation='gelu'):
        super().__init__()
        layers = []
        in_features = input_dim

        if activation.lower() == 'relu':
            self.activ = nn.ReLU()
        elif activation.lower() == 'gelu':
            self.activ = nn.GELU()
        else:
            # Default to GELU if activation is not recognized or add error handling
            self.activ = nn.GELU()
            print(f"Warning: Dense_block activation '{activation}' not recognized. Defaulting to GELU.")

        if hidden_dims: # Ensure hidden_dims is not None and not empty
            for hidden_dim in hidden_dims:
                layers.append(nn.Linear(in_features,hidden_dim))
                layers.append(self.activ)
                in_features = hidden_dim
        # Removed "elif hidden_dims == 0 or "None":" as it's not standard

        layers.append(nn.Linear(in_features,1))
        self.block = nn.Sequential(*layers)

    def forward(self,x):
        return self.block(x)

class MultiModalModel(nn.Module):
    def __init__(self, num_images_per_data, header_mode:str='ffn',
                 image_embeddings_dim_out=None, text_embedding_dim_in=None,
                 text_embedding_dim_out=None, other_features_dim_in=None,
                 other_features_dim_out=None, header_hidden_dims:list=None,
                 nhead:int=8, num_encoder_layers:int=6,num_decoder_layers:int=6,activation:str='gelu'):

        super().__init__()
        """
        header_mode : {'ffn', 'dense', 'transformer','ffn_transformer','dense_transformer'}
        num_images_per_data : Number of images per data point.
        image_embeddings_dim_out : Output dimension for image embeddings.
        text_embedding_dim_in : Input dimension for text embeddings.
        text_embedding_dim_out : Output dimension for text embeddings.
        other_features_dim_in : Input dimension for other features.
        other_features_dim_out : Output dimension for other features.
        header_hidden_dims : Hidden dimensions for the header MLP/Dense block.
        nhead : Number of heads in Transformer.
        num_encoder_layers : Number of encoder layers in Transformer.
        num_decoder_layers : Number of decoder layers in Transformer.
        activation : Activation function to use ('relu' or 'gelu').
        """

        self.header_mode = header_mode
        self.num_images = num_images_per_data

        # Image model (ResNet152)
        self.image_model_weights = torchvision.models.ResNet152_Weights.DEFAULT
        self.image_model = torchvision.models.resnet152(weights=self.image_model_weights)

        if image_embeddings_dim_out is not None:
            self.image_model.fc = nn.Linear(in_features=self.image_model.fc.in_features, out_features=image_embeddings_dim_out)
            self.image_output_dim = image_embeddings_dim_out
        else:
            self.image_output_dim = self.image_model.fc.in_features
            self.image_model.fc = nn.Identity()

        # Text embedding layer
        if text_embedding_dim_out is not None:
            self.text_fc = nn.Linear(in_features=text_embedding_dim_in, out_features=text_embedding_dim_out)
            self.text_embedding_dim = text_embedding_dim_out
        else:
            self.text_fc = nn.Identity()
            self.text_embedding_dim = text_embedding_dim_in

        # Other features embedding layer
        if other_features_dim_out is not None:
            self.rest_feature_fc = nn.Linear(in_features=other_features_dim_in, out_features=other_features_dim_out)
            self.other_features_dim = other_features_dim_out
        else:
            self.rest_feature_fc = nn.Identity()
            self.other_features_dim = other_features_dim_in

        print(f"Image output dimension: {self.image_output_dim}")
        print(f"Text embedding dimension: {self.text_embedding_dim}")
        print(f"Other features dimension: {self.other_features_dim}")

        # Combined input dimension for the header
        # This assumes multiple images are concatenated. If only one image feature vector is used, adjust accordingly.
        self.input_dim = self.image_output_dim * self.num_images + self.text_embedding_dim + self.other_features_dim
        print(f"Combined embedding dimension for header: {self.input_dim}")

        # Activation function
        if activation.lower() == 'relu':
            self.activ = nn.ReLU()
        elif activation.lower() == 'gelu':
            self.activ = nn.GELU()
        else:
            self.activ = nn.GELU() # Defaulting to GELU
            print(f"Warning: MultiModalModel activation '{activation}' not recognized. Defaulting to GELU.")

        # Header layers based on mode
        if self.header_mode.lower() == 'ffn':
            self.header = MLPRegression(self.input_dim, hidden_dims=header_hidden_dims, dropout_prob=0.2, activation=activation)
        elif self.header_mode.lower() == 'dense':
            self.header = Dense_block(self.input_dim, hidden_dims=header_hidden_dims, activation=activation)
        elif self.header_mode.lower() == 'transformer':
            if self.input_dim % nhead != 0:
                raise ValueError(f"Input dimension ({self.input_dim}) must be divisible by nhead ({nhead}) for Transformer.")
            self.transformer_layer = nn.Transformer(d_model=self.input_dim, nhead=nhead,
                                                 num_encoder_layers=num_encoder_layers,
                                                 num_decoder_layers=num_decoder_layers,
                                                 batch_first=True)
            self.header = self.transformer_layer
            self.fc_transformer_out = nn.Linear(self.input_dim, 1) # Final FC layer for transformer output
        elif self.header_mode.lower() == 'ffn_transformer':
            # Intermediate MLP (bottleneck)
            self.bottle_mlp = MLPRegression(self.input_dim, hidden_dims=header_hidden_dims, dropout_prob=0.2, activation=activation)
            # Remove final linear layer from MLP to get bottleneck features
            self.bottle = nn.Sequential(*list(self.bottle_mlp.mlp.children())[:-1])

            # Determine bottleneck output dimension
            # This requires inspecting the actual layers of the MLP.
            # Assuming the last layer before the final linear layer is BatchNorm1d or Dropout, then Linear
            bottleneck_feature_dim = -1
            for layer in reversed(list(self.bottle_mlp.mlp.children())[:-1]):
                if isinstance(layer, nn.Linear):
                    bottleneck_feature_dim = layer.out_features
                    break
            if bottleneck_feature_dim == -1:
                 # Fallback or error if MLPRegression structure changes significantly
                if header_hidden_dims:
                    bottleneck_feature_dim = header_hidden_dims[-1]
                else: # Should not happen if hidden_dims is required by MLPRegression
                    raise ValueError("Cannot determine bottleneck dimension for FFN_Transformer without hidden_dims.")

            if bottleneck_feature_dim % nhead != 0:
                raise ValueError(f"Bottleneck dimension ({bottleneck_feature_dim}) must be divisible by nhead ({nhead}).")
            self.transformer_layer = nn.Transformer(d_model=bottleneck_feature_dim, nhead=nhead,
                                                 num_encoder_layers=num_encoder_layers,
                                                 num_decoder_layers=num_decoder_layers,
                                                 batch_first=True)
            self.header = self.transformer_layer # The transformer itself is part of the header sequence
            self.fc_transformer_out = nn.Linear(bottleneck_feature_dim, 1)
        elif self.header_mode.lower() == 'dense_transformer':
            # Intermediate Dense block (bottleneck)
            self.bottle_dense = Dense_block(self.input_dim, hidden_dims=header_hidden_dims, activation=activation)
            # Remove final linear layer from Dense_block to get bottleneck features
            self.bottle = nn.Sequential(*list(self.bottle_dense.block.children())[:-1])

            bottleneck_feature_dim = -1
            for layer in reversed(list(self.bottle_dense.block.children())[:-1]):
                if isinstance(layer, nn.Linear):
                    bottleneck_feature_dim = layer.out_features
                    break
            if bottleneck_feature_dim == -1:
                if header_hidden_dims:
                    bottleneck_feature_dim = header_hidden_dims[-1]
                else:
                     raise ValueError("Cannot determine bottleneck dimension for Dense_Transformer without hidden_dims.")

            if bottleneck_feature_dim % nhead != 0:
                raise ValueError(f"Bottleneck dimension ({bottleneck_feature_dim}) must be divisible by nhead ({nhead}).")
            self.transformer_layer = nn.Transformer(d_model=bottleneck_feature_dim, nhead=nhead,
                                                 num_encoder_layers=num_encoder_layers,
                                                 num_decoder_layers=num_decoder_layers,
                                                 batch_first=True)
            self.header = self.transformer_layer # The transformer itself is part of the header sequence
            self.fc_transformer_out = nn.Linear(bottleneck_feature_dim, 1)
        else:
            raise ValueError(f"Unsupported header_mode: {self.header_mode}")

    def forward(self, images, text, other_features):
        # Process image embeddings
        # Expects 'images' to be a list of image tensors if num_images > 1
        # or a single batch of image tensors if num_images == 1
        if self.num_images > 0 :
            if isinstance(images, list): # Handles list of image tensors from dataloader
                 # Common case from original files: list of N images per batch item, each (C,H,W)
                 # We need to process each image in the list with the image_model
                 # then stack and flatten
                processed_image_embeddings = [self.activ(self.image_model(img)) for img in images]
                image_embeddings_combined = torch.stack(processed_image_embeddings, dim=1) # (B, num_images, EmbedDim)
                image_embeddings = torch.flatten(image_embeddings_combined, start_dim=1) # (B, num_images * EmbedDim)

            else: # Assumes images is already a tensor (B, num_images, C, H, W) or similar
                  # This path needs careful handling based on actual dataloader output.
                  # For simplicity, adopting the list processing logic first.
                  # If images is (B, C, H, W) for single image:
                if images.ndim == 4 and self.num_images == 1:
                    image_embeddings_combined = self.activ(self.image_model(images)) # (B, EmbedDim)
                     # No need to flatten further unless it's wrapped in a list/stacked again
                    image_embeddings = image_embeddings_combined
                elif images.ndim == 5 and images.shape[1] == self.num_images: # (B, num_images, C, H, W)
                    # Reshape to (B*num_images, C, H, W) for batch processing by resnet
                    b, n, c, h, w = images.shape
                    img_reshaped = images.view(b*n, c, h, w)
                    processed_reshaped = self.activ(self.image_model(img_reshaped))
                    # Reshape back to (B, N, EmbedDim)
                    image_embeddings_combined = processed_reshaped.view(b, n, -1)
                    image_embeddings = torch.flatten(image_embeddings_combined, start_dim=1)
                else:
                    # Fallback or error for unexpected image tensor shape
                    # This was the most common pattern in the provided files:
                    processed_image_embeddings = [self.activ(self.image_model(img)) for img in images] # images is a list of tensors
                    image_embeddings_combined = torch.stack(processed_image_embeddings, dim=1)
                    image_embeddings = torch.flatten(image_embeddings_combined, start_dim=1)

        else: # No images
            image_embeddings = torch.empty(images.shape[0], 0, device=images.device)


        text_embeddings = self.activ(self.text_fc(text))
        other_embeddings = self.activ(self.rest_feature_fc(other_features))

        combined_embeddings = torch.cat((image_embeddings, text_embeddings, other_embeddings), dim=1)

        # Apply activation to combined embeddings before header
        y = self.activ(combined_embeddings)

        # Header processing
        if self.header_mode.lower() in ['ffn', 'dense']:
            y = self.header(y)
        elif self.header_mode.lower() == 'transformer':
            # Transformer expects (batch, seq_len, features)
            # For current setup, seq_len is 1 as we process a single combined vector
            y = y.unsqueeze(1) # Add sequence dimension
            y = self.header(y, y) # Self-attention: src and tgt are the same
            y = y.squeeze(1) # Remove sequence dimension
            y = self.activ(y) # Activation after transformer
            y = self.fc_transformer_out(y) # Final linear layer
        elif self.header_mode.lower() in ['ffn_transformer', 'dense_transformer']:
            y = self.bottle(y) # Pass through bottleneck
            y = y.unsqueeze(1) # Add sequence dimension for transformer
            y = self.header(y, y) # Pass through transformer (self.header is the transformer_layer here)
            y = y.squeeze(1) # Remove sequence dimension
            y = self.activ(y) # Activation after transformer
            y = self.fc_transformer_out(y) # Final linear layer

        return y
