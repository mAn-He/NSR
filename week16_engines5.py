from torch.utils.tensorboard import SummaryWriter
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
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
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

def train(model,optimizer,criterion,eval_metric,train_dataloader, test_dataloader,writer=None, scheduler=None,device='cpu',num_epochs=50,dir_path='model_state_dict'):

    os.makedirs(dir_path,exist_ok=True)
    model.to(device)

    train_eval_list = []
    train_loss_list = []
    test_eval_list = []
    test_loss_list = []
    best_score = float('inf')
    
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")

        # Train
        model.train()

        for param in model.image_model.parameters():
            param.requires_grad = False

        for param in model.image_model.fc.parameters():
            param.requires_grad = True
        
        train_loss = 0.0
        train_eval = 0.0
        for i, batch in tqdm(enumerate(train_dataloader),desc='Training',leave=False):
            images, text_embeddings, other_embeddings, targets = [x.to(device) for x in batch]
            outputs = model(images, text_embeddings, other_embeddings)

            loss = criterion(outputs.flatten(),targets.flatten())
            eval_score = eval_metric(outputs.flatten(), targets.flatten())
            
            if torch.isnan(outputs).any():
                print(f"NaN detected in Train outputs at epoch {epoch+1}, batch {i+1}")
                continue
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_eval +=  eval_score.item()
            train_loss += loss.item()

        train_eval_list.append(train_eval/len(train_dataloader))
        train_loss_list.append(train_loss/len(train_dataloader))

        model.eval()
        test_eval = 0.0
        test_loss = 0.0

        with torch.inference_mode():
            for i, batch in tqdm(enumerate(test_dataloader),desc='Testing',leave=False):
                images,text_embeddings, other_embeddings,targets = [x.to(device) for x in batch]
                outputs = model(images, text_embeddings, other_embeddings)
                
                eval_score = eval_metric(outputs.flatten(), targets.flatten())
                loss = criterion(outputs.flatten(), targets.flatten())
                
                if torch.isnan(outputs).any():
                    print(f"NaN detected in Test outputs at epoch {epoch+1}, batch {i+1}")
                    continue
                
                test_eval += eval_score.item()
                test_loss += loss.item()
       
        test_eval_list.append(test_eval/len(test_dataloader))
        test_loss_list.append(test_loss/len(test_dataloader))

        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss : {train_loss / len(train_dataloader):.4f}, Test Loss : {test_loss / len(test_dataloader):.4f}' )

        print(f'Epoch {epoch+1}/{num_epochs}, Train Eval : {train_eval / len(train_dataloader):.4f}, Test Eval : {test_eval / len(test_dataloader):.4f}' )

        if writer:
            writer.add_scalar(f'Loss/train',train_loss/len(train_dataloader),epoch+1)
            writer.add_scalar(f'Loss/test',test_loss/len(test_dataloader),epoch+1)
            writer.add_scalar(f'Eval/train',train_eval/len(train_dataloader),epoch+1)
            writer.add_scalar(f'Eval/test',test_eval/len(test_dataloader),epoch+1)

        
        if torch.isnan(outputs).any() != True:
            if test_loss/len(test_dataloader) < best_score:
                best_score = test_loss/len(test_dataloader)
                torch.save(model.state_dict(), dir_path+'/'+'model.pt')
                best_model = model
                print(f"Saved best model at epoch {epoch + 1} with Target Loss: {best_score:.4f}")
       
        if scheduler:
            scheduler.step()
    
        plt.figure(figsize=(10, 5))
        plt.plot(range(len(train_eval_list)), train_eval_list, label='Train Eval Score')
        plt.xlabel('Epochs')
        plt.ylabel('Eval Score')
        plt.legend()
        plt.savefig(dir_path+'/'+'train_eval_plot.png')
        plt.close()
        
        plt.figure(figsize=(10, 5))
        plt.plot(range(len(test_eval_list)), test_eval_list, label='Test Eval Score')
        plt.xlabel('Epochs')
        plt.ylabel('Eval Score')
        plt.legend()
        plt.savefig(dir_path+'/'+'test_eval_plot.png')
        plt.close()
        
        plt.figure(figsize=(10, 5))
        plt.plot(range(len(train_loss_list)), train_loss_list, label='Train Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(dir_path+'/'+'train_loss_plot.png')
        plt.close()
        
        plt.figure(figsize=(10, 5))
        plt.plot(range(len(test_loss_list)), test_loss_list, label='Test Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(dir_path+'/'+'test_loss_plot.png')
        plt.close()

    if writer:
        writer.close()
    
    return best_model


def test(model,dataset,dataloader,df,df_true,criterion,eval_metric,device,output_dir=str):

    model.to(device)
    outputs_list = []
    test_loss = 0.0
    test_eval = 0.0
    model.eval()
    with torch.inference_mode():
        for i,batch in tqdm(enumerate(dataloader),desc='Predictiong...',leave=False):
            images,text_embeddings,other_embeddings,targets = [x.to(device) for x in batch]
            outputs = model(images,text_embeddings,other_embeddings)

            
            loss = criterion(outputs.flatten(), targets.flatten())
            eval_score = eval_metric(outputs.flatten(), targets.flatten()) 
            outputs_list.extend(outputs.detach().cpu().numpy())
            test_loss += loss.item()
            test_eval += eval_score.item()
    test_loss /= len(dataloader)
    test_eval /= len(dataloader)
    
    outputs_list = [np.round(output.item(),2) for output in outputs_list]
   
    df['판매수량_pred'] = outputs_list
    saved_path = os.path.join(output_dir,dataset.split('/')[-1].split('.')[0]+'_preds.csv')
    
    hi = pd.DataFrame()
    hi['상품코드'] = df['상품코드']
    hi = hi.merge(df.groupby('상품코드')['판매수량_true'].sum(),on='상품코드')
    hi = hi.merge(df.groupby('상품코드')['판매수량_pred'].sum(),on='상품코드')
    
    test_eval_ItemCode = eval_metric(torch.tensor(hi['판매수량_pred'].values),torch.tensor(hi['판매수량_true'].values)).item()
    test_loss_ItemCode = criterion(torch.tensor(hi['판매수량_pred'].values),torch.tensor(hi['판매수량_true'].values)).item()
    
    loss_info_text = dataset.split('/')[-1].split('.')[0] + '_loss_info.txt'
    loss_info_path = os.path.join(output_dir, loss_info_text )
        
    hi.to_csv(os.path.join(output_dir,dataset.split('/')[-1].split('.')[0]+'_상품코드별_preds.csv'),encoding='UTF-8')
    df.to_csv(os.path.join(output_dir,dataset.split('/')[-1].split('.')[0]+'_preds.csv'),encoding='UTF-8')
    print(f'predict result saved in {saved_path}')

    dict_item_loss = calculate_ad_smape_grouped_by(hi,df_true,criterion)
    dict_item_eval = calculate_ad_smape_grouped_by(hi,df_true,eval_metric)
    
    dict_item_color_loss = calculate_ad_smape_grouped_by(df,df_true,criterion)
    dict_item_color_eval = calculate_ad_smape_grouped_by(df,df_true,eval_metric)
    
    with open(loss_info_path, 'w') as f:
        f.write(f'Eval Score : {test_loss:.4f}\n')
        f.write(f'Loss Score : {test_eval:.4f}\n')
        f.write(f'상품코드별 Eval Score : {test_eval_ItemCode:.4f}\n')
        f.write(f'상품코드별 Loss Score : {test_loss_ItemCode:.4f}\n')
        for k,v in dict_item_eval.items():
            f.write(f'{k} 상품코드별 Eval Score : {v:.4f}\n')

        for k,v in dict_item_loss.items():
            f.write(f'{k} 상품코드별 Loss Score : {v:.4f}\n')
            
        for k,v in dict_item_color_eval.items():
            f.write(f'{k} 상품코드+색상별 Eval Score : {v:.4f}\n')

        for k,v in dict_item_color_loss.items():
            f.write(f'{k} 상품코드+색상별 Eval Loss : {v:.4f}\n')

def calculate_adjusted_smape(row, loss_fn):
    y_true = torch.tensor([row['판매수량_true']], dtype=torch.float32)
    y_pred = torch.tensor([row['판매수량_pred']], dtype=torch.float32)
    score = loss_fn(y_pred, y_true).item()
    return score

    
def run(model,dataset,optimizer,criterion,eval_metric,train_dataloader,test_dataloader,dir_path,df,df_true,scheduler=None,writer=None,device='cpu',num_epochs=150):
    best_model = train(model,optimizer,criterion,eval_metric,train_dataloader,test_dataloader,writer=writer,scheduler=scheduler,device=device,num_epochs=num_epochs,dir_path=dir_path)
    print('Best Model for Train dataset')
    test(model=best_model,dataset=dataset[0],dataloader=train_dataloader,df=df[0],df_true=df_true[0],criterion=criterion,eval_metric=eval_metric,device=device,output_dir=dir_path)
    print('Best Model for Test dataset')
    test(model=best_model,dataset=dataset[1],dataloader=test_dataloader,df=df[1],df_true=df_true[1],criterion=criterion,eval_metric=eval_metric,device=device,output_dir=dir_path)