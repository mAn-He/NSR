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
from engines.week16_engines4 import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import boxcox
from scipy.special import inv_boxcox,boxcox1p,inv_boxcox1p

class MultiModalModel(nn.Module):
    def __init__(self, num_images_per_data, header_mode:str='ffn',
                 image_embeddings_dim_out=None, text_embedding_dim_in=None, 
                 text_embedding_dim_out=None, other_features_dim_in=None, 
                 other_features_dim_out=None, header_hidden_dims:list=None,
                 nhead:int=8, num_encoder_layers:int=6,num_decoder_layers:int=6,activation:str='gelu'):
        
        super().__init__()
        """
        header_mode : {'ffn', 'dense', 'transformer','ffn_transformer','dense_transformer'} 이 중에서 선택해주세요.
        num_images_per_data : 데이터 포인트 당 이미지 갯수 (데이터 로드 하는 과정에서 1개 가진 이미지도 복제 하여 모두 2개로 만들어 놨습니다. 수정하실 필요 없이 2로 고정하시면 됩니다.)
        image_embeddings_dim_out : 원하는 이미지 emedding dimension (맘대로 가능)
        text_embedding_dim_in : dataframe의 텍스트 칼럼 dimension (데이터 프레임에 있는 텍스트 임베딩 칼럼 갯수.. 고정값)
        text_embedding_dim_out : 선형 변환을 통한 텍스트 embedding dim (맘대로 가능)
        other_features_dim_in : dataframe의 텍스트 embedding 칼럼 및 이미지, target 칼럼 제외, 나머지 feature 칼럼들 dimension (데이터 프레임에 있는 텍스트 제외한 칼럼들 갯수.. 고정값)
        other_features_dim_out : 선형 변환을 통한 output feature dim (맘대로 가능)
        header_hidden_dims : 각 모델의 ffn이나, dense 블록의 hidden dim
        nhead : transformer 계열 모델에서 num head,
        num_encoder_layers : transformer의 encoder layer 갯수
        num_decoder_layers : transformer의 decoder layer 갯수
        """
        
        # Image
        self.header_mode = header_mode
        self.num_images = num_images_per_data
        self.image_model_weights = torchvision.models.ResNet152_Weights.DEFAULT
        self.image_model = torchvision.models.resnet152(weights=self.image_model_weights)
        
        if image_embeddings_dim_out is not None:
            self.image_model.fc = nn.Linear(in_features=self.image_model.fc.in_features, out_features=image_embeddings_dim_out)
            self.image_output_dim = image_embeddings_dim_out
        else:
            self.image_output_dim = self.image_model.fc.in_features
            self.image_model.fc = nn.Identity()

        # Text
        if text_embedding_dim_out is not None:
            self.text_fc = nn.Linear(in_features=text_embedding_dim_in, out_features=text_embedding_dim_out)
            self.text_embedding_dim = text_embedding_dim_out
        else:
            self.text_fc = nn.Identity()
            self.text_embedding_dim = text_embedding_dim_in

        # 나머지 feature
        if other_features_dim_out is not None:
            self.rest_feature_fc = nn.Linear(in_features=other_features_dim_in, out_features=other_features_dim_out)
            self.other_features_dim = other_features_dim_out
        else:
            self.rest_feature_fc = nn.Identity()
            self.other_features_dim = other_features_dim_in

        print('image_output_dim :',self.image_output_dim)
        print('text_embedding_dim :',self.text_embedding_dim)
        print('other_features_dim :',self.other_features_dim)
        # 막단 layer
        self.input_dim = self.image_output_dim * self.num_images + self.text_embedding_dim + self.other_features_dim
        
        self.head_mlp = MLPRegression(self.input_dim, hidden_dims=header_hidden_dims, dropout_prob=0.2, activation=activation)
        
        if activation.lower() == 'relu':
            self.activ = nn.ReLU()
        elif activation.lower() == 'gelu':
            self.activ = nn.GELU()
        
        self.dense_block = Dense_block(self.input_dim,hidden_dims=header_hidden_dims,activation=activation)
        
        print('concated embed dim :',self.input_dim)

        if self.header_mode == 'transformer':
            if self.input_dim % nhead != 0:
        
                raise ValueError(f"concated embed dim (= {self.image_output_dim * self.num_images + self.text_embedding_dim + self.other_features_dim}) 은 nhead로 나누어 떨어져야만 합니다")
            
            else:
                self.transformer = nn.Transformer(d_model = self.input_dim,
                                          nhead = nhead,
                                          num_encoder_layers = num_encoder_layers,
                                          num_decoder_layers = num_decoder_layers,
                                          batch_first=True)
                self.fc = nn.Linear(self.input_dim,1)
        # if self.header_mode.lower() == 'ffn':
        #     self.header = self.head_mlp

        # elif self.header_mode.lower() == 'dense':
        #     self.header = self.dense_block

        # elif self.header_mode.lower() == 'transformer':
        
            

        

        if self.header_mode.lower() == 'ffn_transformer':
            self.bottle = nn.Sequential(*list(*self.head_mlp.children())[:-1])
            self.bottle_out_dim = list(self.bottle)[-4].out_features
            if self.bottle_out_dim % nhead != 0:
        
                raise ValueError(f"Bottle Neck Output Dim (={self.bottle_out_dim}) 은 nhead로 나누어 떨어져야만 합니다")
            
            else:
                self.transformer = nn.Transformer(d_model = self.bottle_out_dim,
                                              nhead = nhead,
                                              num_encoder_layers = num_encoder_layers,
                                              num_decoder_layers = num_decoder_layers,
                                              batch_first=True)
                self.fc = nn.Linear(self.bottle_out_dim,1)
                
        elif self.header_mode.lower() == 'dense_transformer':
            self.bottle = nn.Sequential(*list(*self.dense_block.children())[:-1])
            self.bottle_out_dim = list(self.bottle)[-1].out_features
            
            if self.bottle_out_dim % nhead != 0:
        
                raise ValueError(f"Bottle Neck Output Dim (={self.bottle_out_dim}) 은 nhead로 나누어 떨어져야만 합니다")
            
            else:
                self.transformer = nn.Transformer(d_model = self.bottle_out_dim,
                                              nhead = nhead,
                                              num_encoder_layers = num_encoder_layers,
                                              num_decoder_layers = num_decoder_layers,
                                              batch_first=True) 
                self.fc = nn.Linear(self.bottle_out_dim,1)
                
    def forward(self, images, text, other_features):
        image_embeddings = [self.activ(self.image_model(image)) for image in images]
        image_embeddings = torch.flatten(torch.stack(image_embeddings, dim=0), start_dim=1)

        text_embeddings = self.text_fc(text)
        text_embeddings = self.activ(text_embeddings)
        
        other_embeddings = self.rest_feature_fc(other_features)
        other_embeddings = self.activ(other_embeddings)
        
        combined_embeddings = torch.cat((image_embeddings, text_embeddings, other_embeddings), dim=1)

        y =self.activ(combined_embeddings)

        if self.header_mode.lower() == 'ffn':
            y = self.head_mlp(y)

        elif self.header_mode.lower() == 'dense':
            y = self.dense_block(y)

        elif self.header_mode.lower() == 'transformer':
            
            # y = y.unsqueeze(1)
            y = self.transformer(y,y)
            y = self.activ(y)
            y = self.fc(y)

        else:

            y = self.bottle(y)
            y = self.transformer(y,y)
            y = self.activ(y)
            y = self.fc(y)
           

        return y

class MLPRegression(nn.Module):
    def __init__(self, input_dim, hidden_dims, dropout_prob=0.2,activation='gelu'):
        super(MLPRegression, self).__init__()
        layers = []
        in_features = input_dim

        if activation.lower() == 'relu':
            self.activ = nn.ReLU()
        elif activation.lower() == 'gelu':
            self.activ = nn.GELU()
            
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
    def __init__(self, input_dim,hidden_dims,activation='gelu'):
        super().__init__()
        layers = []
        in_features = input_dim
        
        if activation.lower() == 'relu':
            self.activ = nn.ReLU()
        elif activation.lower() == 'gelu':
            self.activ = nn.GELU()
            
        if hidden_dims:
            for hidden_dim in hidden_dims:
                layers.append(nn.Linear(in_features,hidden_dim))
                layers.append(self.activ) 
                in_features = hidden_dim

        elif hidden_dims == 0 or "None":
            pass

        layers.append(nn.Linear(in_features,1))
        self.block = nn.Sequential(*layers)

    def forward(self,x):
        return self.block(x)
