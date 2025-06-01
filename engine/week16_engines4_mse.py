import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch.nn as nn
import torchvision
import pandas as pd
from torchvision.models import resnet18
from torchvision import transforms
from PIL import Image
import numpy as np
import ast
from tqdm import tqdm
from scipy.stats import boxcox
from scipy.special import inv_boxcox
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
from scipy.stats import boxcox
from scipy.special import inv_boxcox,boxcox1p,inv_boxcox1p
import torch
import numpy as np
import random

def calculate_ad_smape_grouped_by(df_pred,df_true,eval_metric):

    if '칼라' in df_pred.columns.tolist():
        df_true = df_true[['상품코드','칼라명2','카테고리']]
        df_true = df_true.rename(columns={'칼라명2':'칼라'})
    
        df = df_pred.merge(df_true,how='left',on=['상품코드','칼라'])

    else:
        df_true = df_true[['상품코드','카테고리']]
        df = df_pred.merge(df_true,how='left',on='상품코드')
    
    cats = df['카테고리'].unique()
    dictionary = {}
        
    for cat in cats:

        x = df[df['카테고리'] == cat]

        y1 = torch.tensor(x['판매수량_true'].values,dtype=torch.float)
        y2 = torch.tensor(x['판매수량_pred'].values,dtype=torch.float)
        
        eval_score = eval_metric(y1,y2)
        dictionary[cat] = eval_score

    return dictionary

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
        """

        param epsilon: default 1e-6
        """
        super(AdjustedSMAPELoss, self).__init__()
        self.epsilon = epsilon

    def forward(self, y_pred, y_true):
      
        numerator = torch.abs(y_pred - y_true)
        denominator = torch.abs(y_pred) + torch.abs(y_true) + self.epsilon
        
        
        smape_score = torch.mean(numerator / denominator * 2)/2
        
        return smape_score

class nsr_img_txt_dataset(Dataset):
    def __init__(self, dataframe, transform=None,fixed_num_images=2):
        self.dataframe = dataframe
        self.transform = transform
        self.fixed_num_images = fixed_num_images
    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]

        image_paths = ast.literal_eval(row['이미지파일'])
    
        text_embeddings = row.filter(like='설명').values.astype(np.float32)
        target = row['판매수량']
        target_scaled = row.get('판매수량_scaled',None)
        
        other_embeddings = row.drop(row.filter(like='설명').index,axis=0)
        other_embeddings = other_embeddings.drop(['판매수량','이미지파일','판매수량_scaled'],errors='ignore').values.astype(np.float32)
        
        images = [Image.open(image_path) for image_path in image_paths]
        if self.transform:
            images = [self.transform(image) for image in images]

        while len(images) < self.fixed_num_images:
            images.append(images[-1])
        images = images[:self.fixed_num_images]
        
        images_tensor = torch.stack(images)
        
        text_embeddings_tensor = torch.tensor(text_embeddings,dtype=torch.float32)
        other_embeddings_tensor = torch.tensor(other_embeddings,dtype=torch.float32)
        target_tensor = torch.tensor(target,dtype=torch.float32)

        if target_scaled is not None:
            target_scaled_tensor = torch.tensor(target_scaled,dtype=torch.float32)
        
            return images_tensor, text_embeddings_tensor, other_embeddings_tensor, target_tensor, target_scaled_tensor

        else:
            return images_tensor, text_embeddings_tensor, other_embeddings_tensor, target_tensor

def datapreprocessing(train_df, test_df, dataloader_generator, image_resize_shape=(256, 256), image_normalizing=False,batch_size=64, shuffle=True, num_workers=None, pin_memory=False):
    """
    train_df : DataFrame for Training
    test_df : DataFrame for Testing
    image_resize_shape : image_size to reshape 
    batch_size : Batch size
    """
    import os
    from torchvision import transforms
    from torch.utils.data import DataLoader
    
    assert image_normalizing in [True, False], "image_normalizing must be either True or False"
    
    if image_normalizing == False:
        transform = transforms.Compose([
            transforms.Resize(image_resize_shape),
            transforms.ToTensor()
        ])

    else:
        transform = transforms.Compose([
            transforms.Resize(image_resize_shape),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    
    print(transform)

    train_dataset = nsr_img_txt_dataset(train_df, transform=transform)
    test_dataset = nsr_img_txt_dataset(test_df, transform=transform)

    num_workers = num_workers if num_workers is not None else os.cpu_count()

    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=batch_size,
        shuffle=shuffle,
        generator=dataloader_generator,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    test_dataloader = DataLoader(
        test_dataset, 
        batch_size=batch_size,
        generator=dataloader_generator,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    return train_dataloader, test_dataloader

def train(model,optimizer,criterion, eval_metric,train_dataloader, test_dataloader,loss_ratio:float=1, scheduler=None,device='cpu',num_epochs=50,dir_path='model_state_dict',a=None,b=None,boxcox_lambda=None):

    os.makedirs(dir_path,exist_ok=True)
    model.to(device)

    train_loss_scaled_list=[]
    test_loss_scaled_list=[]
    train_eval_scaled_list = []
    train_eval_unscaled_list = []
    test_eval_scaled_list = []
    test_eval_unscaled_list = []
    best_eval = float('inf')
    best_loss = float('inf')
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")

        # Train
        model.train()

        for param in model.image_model.parameters():
            param.requires_grad = False

        for param in model.image_model.fc.parameters():
            param.requires_grad = True
        
        train_eval_scaled = 0.0
        train_eval_unscaled = 0.0
        train_loss_scaled = 0.0
        train_loss_unscaled = 0.0
        for i, batch in tqdm(enumerate(train_dataloader),desc='Training',leave=False):
            images, text_embeddings, other_embeddings, targets_unscaled, targets_scaled = [x.to(device) for x in batch]
            outputs_scaled = model(images, text_embeddings, other_embeddings)

            loss_scaled = criterion(outputs_scaled.flatten(),targets_scaled)
            eval_scaled = eval_metric(outputs_scaled.flatten(), targets_scaled)
            # outputs_unscaled = torch.pow(((outputs_scaled.flatten() + b)/a),1/boxcox_lambda)
            if torch.isnan(loss_scaled).any():
                print(f"NaN detected in Train Loss at epoch {epoch+1}, batch {i+1}")
                continue
            
            if torch.isnan(outputs_scaled).any() or torch.isnan(targets_scaled).any():
                print(f"NaN detected in Train outputs or targets at epoch {epoch+1}, batch {i+1}")
                continue

            
            # outputs_unscaled = inv_boxcox1p(outputs_scaled.detach().cpu().numpy(),boxcox_lambda)
            # outputs_unscaled = torch.tensor(outputs_unscaled,dtype=torch.float,device=device)
            outputs_unscaled = inv_boxcox_torch(outputs_scaled,a,b,boxcox_lambda)
            
            if torch.isnan(outputs_unscaled).any():
                print(f"NaN detected in Train outputs_unscaled at epoch {epoch+1}, batch {i+1}")
                continue
            
            eval_unscaled = eval_metric(outputs_unscaled.flatten(), targets_unscaled.flatten())
            loss_unscaled = criterion(outputs_unscaled.flatten(), targets_unscaled.flatten())
            
            if torch.isnan(eval_unscaled).any() or torch.isnan(eval_scaled).any() or torch.isnan(loss_unscaled).any():
                print(f"NaN detected in Train at epoch {epoch+1}, batch {i+1}")
                continue

            loss = (loss_ratio) * loss_scaled + (1-loss_ratio) * loss_unscaled
            
            optimizer.zero_grad()
            loss.backward()
        

            optimizer.step()
            train_eval_scaled +=  eval_scaled.item()
            train_loss_scaled += loss_scaled.item()
            train_eval_unscaled += eval_unscaled.item()
            train_loss_unscaled += loss_unscaled.item()
        train_loss_scaled_list.append(train_loss_scaled/len(train_dataloader))
        # train_loss_unscaled_list.append(train_loss_unscaled/len(train_dataloader))
        train_eval_scaled_list.append(train_eval_scaled/len(train_dataloader))
        train_eval_unscaled_list.append(train_eval_unscaled/len(train_dataloader))

        model.eval()
        test_eval_scaled = 0.0
        test_eval_unscaled = 0.0
        test_loss_scaled = 0.0
        test_loss_unscaled = 0.0
        with torch.inference_mode():
            for i, batch in tqdm(enumerate(test_dataloader),desc='Testing',leave=False):
                images,text_embeddings, other_embeddings, targets_unscaled, targets_scaled = [x.to(device) for x in batch]
                outputs_scaled = model(images, text_embeddings, other_embeddings)
                eval_scaled = eval_metric(outputs_scaled, targets_scaled)
                loss_scaled = criterion(outputs_scaled, targets_scaled)
             
                # outputs_unscaled = inv_boxcox1p(outputs_scaled.detach().cpu().numpy(),boxcox_lambda)
                # outputs_unscaled = torch.tensor(outputs_unscaled,dtype=torch.float,device=device)
                outputs_unscaled = inv_boxcox_torch(outputs_scaled,a,b,boxcox_lambda)
                # print('ouputs_unscaled.shape:',outputs_unscaled.shape)
                # outputs_unscaled = torch.tensor(outputs_unscaled,dtype=torch.float,device=device)
                
                if torch.isnan(outputs_unscaled).any():
                    print(f"NaN detected in test outputs_unscaled at epoch {epoch+1}, batch {i+1}")
                    continue
                
                eval_unscaled = eval_metric(outputs_unscaled.flatten(), targets_unscaled.flatten())
                loss_unscaled = criterion(outputs_unscaled.flatten(), targets_unscaled.flatten())
                
                if torch.isnan(loss_unscaled).any() or torch.isnan(loss_scaled).any():
                    print(f"NaN detected in Test loss at epoch {epoch+1}, batch {i+1}")
                    continue
                
                test_eval_scaled += eval_scaled.item()
                test_eval_unscaled += eval_unscaled.item()
                test_loss_scaled += loss_scaled.item()
                test_loss_unscaled += loss_unscaled.item()

        test_loss_scaled_list.append(test_loss_scaled/len(test_dataloader))
        test_eval_scaled_list.append(test_eval_scaled/len(test_dataloader))
        test_eval_unscaled_list.append(test_eval_unscaled/len(test_dataloader))

        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss Scaled: {train_loss_scaled / len(train_dataloader):.4f}, Test Loss Scaled: {test_loss_scaled / len(test_dataloader):.4f}' )

        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss UnScaled: {train_loss_unscaled / len(train_dataloader):.4f}, Test Loss UnScaled: {test_loss_unscaled / len(test_dataloader):.4f}' )
        
        print(f'Epoch {epoch+1}/{num_epochs}, Train Eval Scaled: {train_eval_scaled / len(train_dataloader):.4f}, Test Eval Scaled: {test_eval_scaled / len(test_dataloader):.4f}')
        
        print(f'Epoch {epoch+1}/{num_epochs}, Train Eval UnScaled: {train_eval_unscaled / len(train_dataloader):.4f}, Test Eval UnScaled: {test_eval_unscaled / len(test_dataloader):.4f}')

        # if torch.isnan(outputs_unscaled).any() != True:
        #     if test_eval_unscaled/len(test_dataloader) < best_eval:
        #         best_eval = test_eval_unscaled/len(test_dataloader)
        #         torch.save(model.state_dict(), dir_path+'/'+'model.pt')
        #         print(f"Saved best model at epoch {epoch + 1} with Unscaled Target Eval: {best_eval:.4f}")
        if torch.isnan(outputs_unscaled).any() != True:
            if test_loss_scaled/len(test_dataloader) < best_loss:
                best_loss = test_loss_scaled/len(test_dataloader)
                print(test_loss_scaled)
                torch.save(model.state_dict(), dir_path+'/'+'model.pt')
                print(f"Saved best model at epoch {epoch + 1} with Scaled Target Loss: {best_loss:.4f}")


            
        if scheduler:
            scheduler.step()

        plt.figure(figsize=(10, 5))
        plt.plot(range(len(train_eval_scaled_list)), train_eval_scaled_list, label='Train Eval UnNormalized')
        plt.xlabel('Epochs')
        plt.ylabel('Eval Score')
        plt.legend()
        plt.savefig(dir_path+'/'+'train_eval_unnormalized_plot.png')
        plt.close()
        
        plt.figure(figsize=(10, 5))
        plt.plot(range(len(test_eval_scaled_list)), test_eval_scaled_list, label='Test Eval Scaled')
        plt.xlabel('Epochs')
        plt.ylabel('Eval Score')
        plt.legend()
        plt.savefig(dir_path+'/'+'test_eval_unnormalized_plot.png')
        plt.close()
        
        
        plt.figure(figsize=(10, 5))
        plt.plot(range(len(train_eval_unscaled_list)), train_eval_unscaled_list, label='Train Eval UnNormalized')
        plt.xlabel('Epochs')
        plt.ylabel('Eval Score')
        plt.legend()
        plt.savefig(dir_path+'/'+'test_eval_unnormalized_plot.png')
        plt.close()
       
        plt.figure(figsize=(10, 5))
        plt.plot(range(len(test_eval_unscaled_list)), test_eval_unscaled_list, label='Test Eval UnNormalized')
        plt.xlabel('Epochs')
        plt.ylabel('Eval Score')
        plt.legend()
        plt.savefig(dir_path+'/'+'test_eval_unnormalized_plot.png')
        plt.close()
        
        plt.figure(figsize=(10, 5))
        plt.plot(range(len(train_loss_scaled_list)), train_loss_scaled_list, label='Train loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(dir_path+'/'+'train_normalized_loss.png')
        plt.close()
        
        plt.figure(figsize=(10, 5))
        plt.plot(range(len(test_loss_scaled_list)), test_loss_scaled_list, label='Test loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(dir_path+'/'+'test_normalized_loss.png')
        plt.close()
        
    # return train_losses_scaled, test_losses_scaled, train_losses_unscaled, test_losses_unscaled

def boxcox_standardscaling(df_train,df_test):
    from sklearn.preprocessing import StandardScaler
    from scipy.stats import boxcox
    from scipy.special import inv_boxcox
    

    transformed_df_train, optimal_lambda = boxcox(df_train['판매수량'])
    transformed_df_test = boxcox(df_test['판매수량'],lmbda=optimal_lambda)
    scaler = StandardScaler()
    scaled_df_train = scaler.fit_transform(transformed_df_train.reshape(-1,1))
    scaled_df_test = scaler.transform(transformed_df_test.reshape(-1,1))
    df_train['판매수량_scaled'] = scaled_df_train
    df_test['판매수량_scaled'] = scaled_df_test
    return df_train, df_test, scaler, optimal_lambda

def inv_boxcox_torch(x, a, b,lmbda):
    epsilon = 1e-6  # 작은 값을 더해서 오버플로우 방지
    if lmbda == 0:
        return torch.exp(x)
    else:
        return torch.pow((x + b)/a, 1 / lmbda)

def inv_scaling(x, scaler, optimal_lambda, device='cpu'):
    x = torch.tensor(x, dtype=torch.float32, device=device)
    x_unscaled = scaler.inverse_transform(x.cpu().numpy().reshape(-1, 1)).flatten()
    x_unscaled = torch.tensor(x_unscaled, dtype=torch.float32, device=device)
    x_unscaled = inv_boxcox_torch(x_unscaled, a,b,optimal_lambda)
    # print('x_unscaled_boxcox_inv:',x_unscaled)
    return x_unscaled


def train2(model,optimizer,criterion,eval_metric,train_dataloader, test_dataloader, scheduler=None,device='cpu',
          num_epochs=50,dir_path='model_state_dict',a=None,b=None,boxcox_lambda=None):

    os.makedirs(dir_path,exist_ok=True)
    model.to(device)

    train_losses = []
    test_losses = []
    best_loss = float('inf')
    
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")

        # Train
        model.train()

        for param in model.image_model.parameters():
            param.requires_grad = False

        for param in model.image_model.fc.parameters():
            param.requires_grad = True
        
        train_loss = 0.0
        for i, batch in tqdm(enumerate(train_dataloader),desc='Training',leave=False):
            images, text_embeddings, other_embeddings, targets_unscaled, targets_scaled = [x.to(device) for x in batch]
            outputs = model(images, text_embeddings, other_embeddings)

            loss = criterion(outputs.flatten(), targets_unscaled)
            optimizer.zero_grad()
            loss.backward()

            optimizer.step()
            train_loss +=  loss.item()

        train_losses.append(train_loss/len(train_dataloader))

        model.eval()
        test_loss = 0.0
        with torch.inference_mode():
            for i, batch in tqdm(enumerate(test_dataloader),desc='Testing',leave=False):
                images,text_embeddings, other_embeddings, targets_unscaled, targets_scaled = [x.to(device) for x in batch]
                outputs = model(images, text_embeddings, other_embeddings)
                loss = eval_metric(outputs, targets_unscaled)
                test_loss += loss.item()

        test_losses.append(test_loss/len(test_dataloader))

        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss : {train_loss / len(train_dataloader):.4f}, Test Loss : {test_loss / len(test_dataloader):.4f}')

        if test_loss/len(test_dataloader) < best_loss:
            best_loss = test_loss/len(test_dataloader)
            torch.save(model.state_dict(), dir_path+'/'+'model.pt')
            print(f"Saved best model at epoch {epoch + 1} with Unscaled Target Loss: {best_loss:.4f}")
       
        if scheduler:
            scheduler.step()

        plt.figure(figsize=(10, 5))
        plt.plot(range(len(train_losses)), train_losses, label='Train Loss')
        plt.plot(range(len(test_losses)), test_losses, label='Test Loss ')
    
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(dir_path+'/'+'training_loss_plot.png')

    return train_losses, test_losses