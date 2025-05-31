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
from engines.week16_engines4_mse_knn_trial import *
# from engines.models_transformer import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import boxcox
from scipy.special import inv_boxcox,boxcox1p,inv_boxcox1p



def none_or_int(value):
    if value.lower() == 'none':
        return None
    return int(value)

parser = argparse.ArgumentParser(description='Training script for MultiModalMOdel')

parser.add_argument('--device',type=int,default=0,help='device number for gpu accelerating (default = 0)')
parser.add_argument('--batch_size',type=int,default=64,help='batch size (default=64)')
parser.add_argument('--image_embeddings_dim_out', type=none_or_int, default=128, help='Add Linear Layer, Output dimension for image embeddings (default=128, set "None" for no linear embedding proj)')
parser.add_argument('--text_embeddings_dim_out', type=none_or_int, default=128, help='Add Linear Layer, Output dimension for text embeddings (default=128, set "None" for no linear embedding proj))')
parser.add_argument('--other_features_dim_out', type=none_or_int, default=None, help='Add Linear Layer, Output dimension for other features embeddings (default=128, set "None" for no linear embedding proj))')
parser.add_argument('--header_mode',type=str,default='FFN',help='FFN, Dense, Transformer, FFN_Transformer, Dense_Transformer')
parser.add_argument('--header_hidden_dims', type=int, nargs='+', default=[128], help='Add Linear Layer, Hidden dimensions for the header (default=128)')
parser.add_argument('--dir_path', type=str, default='this_is_experiment', help='Directory path to save the model state dict (default="model_state_dict")') 
parser.add_argument('--train_dataset',type=str, default='/home/sflab/SFLAB/sungheon/nsr/dataset/nsr_train_할인율0_중분류.csv',help='train dataset path (default="../dataset/nsr_train.csv")')
parser.add_argument('--test_dataset',type=str, default='/home/sflab/SFLAB/sungheon/nsr/dataset/nsr_test_할인율0_중분류.csv',help='test dataset path (default="../dataset/nsr_test.csv")')
parser.add_argument('--random_seed',type=int, default=42,help='random seed (default=42)')
parser.add_argument('--image_size',type=int,default=256,help='image size for transforming (default=256)')
parser.add_argument('--learning_rate',type=float,default=0.01,help='learning rate (default=0.0000001)')
parser.add_argument('--num_epochs',type=int,default=50,help='Num epochs for training (default=50)')
parser.add_argument('--loss_ratio',type=float,default=1,help='Scaled/Unscaled Loss ratio, default=1(totally weight on target Scaled loss (default=1)')
parser.add_argument('--image_normalizing',type=bool,default=False,help='Decide wheter use image nomralizing or not(default=False)')
parser.add_argument('--nhead',type=int,default=4,help='transformer nhead')
parser.add_argument('--num_encoder_layers',type=int,default=6,help='transformer num encoder layers')
parser.add_argument('--num_decoder_layers',type=int,default=6,help='transformer num decoder layers')
parser.add_argument('--model_path',type=str,default='engines/models_transformer_knn_trial.py',help='model.py path')
parser.add_argument('--activation_func',type=str,default='gelu',help='gelu or relu')

if '--help' in sys.argv or '-h' in sys.argv:
    parser.print_help()
    sys.exit()
    
if 'ipykernel' in sys.modules:
    # Jupyter 노트북에서 실행 중인 경우
    args = parser.parse_args(args=[])
else:
    # 일반 스크립트로 실행 중인 경우
    args = parser.parse_args()

model_path = args.model_path
model_py = model_path.split('/')[-1]
if model_py == 'models_transformer.py':
    from engines.models_transformer import *
elif model_py == 'models_transformer_v2.py':
    from engines.models_transformer_v2 import *
elif model_py == 'models_transformer_knn_trial.py':
    from engines.models_transformer_knn_trial import *

set_random_seed(args.random_seed)

os.makedirs(args.dir_path, exist_ok=True)
config_path = os.path.join(args.dir_path, 'config.json')
with open(config_path, 'w',encoding='utf-8') as f:
    json.dump(vars(args), f, indent=4,ensure_ascii=False)

df_train = pd.read_csv(args.train_dataset)
df_train = df_train.drop(columns=['Unnamed: 0', '판매시작연도', '판매첫날', '상품코드', '판매일자', '상품명', '상품명2', '칼라', '칼라명', '칼라명2', '현재가', '할인율(%)', '파일경로', '이미지갯수', '외관설명', '기능설명','카테고리'],errors='ignore')

cols = df_train.columns.tolist()
df_test = pd.read_csv(args.test_dataset)
df_test = df_test[cols]
df_train = df_train.drop_duplicates()
df_test = df_test.drop_duplicates()

y_train = np.array(df_train['판매수량'].tolist())
y_test = np.array(df_test['판매수량'].tolist())

# y_train = y_train +1
# y_test = y_test + 1

boxcox_y_train, lambda_train = boxcox(y_train)

lowerbound = np.quantile(y_train, 0.16)
upperbound = np.quantile(y_train, 0.84)

a = 2 / (lowerbound**lambda_train - upperbound**lambda_train)
b = a * (lowerbound**lambda_train) - 1

y_train_boxcoxed_with_shifting = a * (y_train**lambda_train) - b
y_test_boxcoxed_with_shifting = a * (y_test**lambda_train) - b
# df_train['판매수량_scaled'] = boxcox_y_train
# df_test['판매수량_scaled'] = boxcox_y_test
df_train['판매수량_scaled'] = y_train_boxcoxed_with_shifting
df_test['판매수량_scaled'] = y_test_boxcoxed_with_shifting

# set random seed
generator = torch.Generator()
generator.manual_seed(args.random_seed)

train_dataloader, test_dataloader = datapreprocessing(df_train, df_test, generator, image_resize_shape=(args.image_size, args.image_size), image_normalizing=args.image_normalizing,batch_size=args.batch_size, num_workers=os.cpu_count(), pin_memory=True)

# define embdding dims
text_embedding_dim_in = len(df_train.filter(like='설명').columns)
other_features_dim_in = df_train.drop(df_train.filter(like='설명').columns,axis=1).shape[-1]-3


model = MultiModalModel(num_images_per_data=2,
                        header_mode= args.header_mode,
                        image_embeddings_dim_out=args.image_embeddings_dim_out,
                        text_embedding_dim_in=text_embedding_dim_in,
                        text_embedding_dim_out=args.text_embeddings_dim_out,
                        other_features_dim_in=other_features_dim_in,
                        other_features_dim_out=args.other_features_dim_out,
                        header_hidden_dims=args.header_hidden_dims,
                        nhead = args.nhead,
                        num_encoder_layers = args.num_encoder_layers,
                        num_decoder_layers = args.num_decoder_layers)

# set hyperparameters

optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay= 0.1)
lr_scheduler = optim.lr_scheduler.StepLR(optimizer,step_size=5,gamma=0.2)


criterion = nn.MSELoss()
eval_metric = AdjustedSMAPELoss()
device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')

train(model,optimizer,criterion,eval_metric,train_dataloader,test_dataloader,loss_ratio=args.loss_ratio,device=device,num_epochs=args.num_epochs,dir_path=args.dir_path,a=a,b=b,boxcox_lambda=lambda_train)