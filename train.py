from random import shuffle
import time
from turtle import distance
from torch.utils.data import Dataset, DataLoader
from torch import nn
from torch import sigmoid
import torch
import sys
import gc
import os
from tqdm import tqdm
import torch
class Trainer:
    def __init__(self,model,data,device,model_name,save_step,save_path,batch_size,max_epoch,init_lr):
        self.model=model.to(device)
        print(self.model)
        self.train_data=data
        self.device=device
        self.model_name=model_name
        self.save_step=save_step
        self.save_path=save_path
        self.batch_size=batch_size
        self.max_epoch=max_epoch
        
        self.optimizer=torch.optim.Adam(self.model.parameters(),lr=init_lr,betas=(0.9,0.999),weight_decay=0.1,maximize=False)
        def _make_torch_loader():
            train_data_total=len(self.train_data)
            train_len_setting=int(train_data_total*90/100)
            val_len_setting=train_data_total-train_len_setting
            self.val_set=self.train_data.SplitSet(train_len_setting,random_offset=False)
            self.train_data.prepare_train()
            self.val_set.prepare_train()
            train_loader=DataLoader(self.train_data,batch_size=self.batch_size,shuffle=True,pin_memory=True)
            val_loader=DataLoader(self.val_set,batch_size=self.batch_size,shuffle=True,pin_memory=True)
            del self.train_data,self.val_set
            gc.collect()
            self.train_data=train_loader
            self.val_data=val_loader
            print("find {} train logs, {} validation logs".format(train_len_setting,val_len_setting))
        _make_torch_loader()
    
    def start_train(self,using_epoch=None):
        if using_epoch is None:
            using_epoch=self.max_epoch
        assert(using_epoch>1)
        self.best_loss=1e10

        for epoch in range(0,using_epoch):
            self.train(epoch=epoch)
            if epoch>=using_epoch//2 and epoch%2==0:
                self.valid(epoch)
                if epoch % self.save_step==0:
                    self._save_checkpoint(epoch,True,"epoch_{}".format(epoch))
        self._save_checkpoint(epoch,True,'last')
    def _save_checkpoint(self,epoch,save_optimizer=True,suffix=''):
        checkpoint={
            "epoch":epoch,
            "state_dict":self.model.state_dict(),
            "best_loss":self.best_loss
        }
        if save_optimizer:
            checkpoint["optimizer"]=self.optimizer.state_dict()
        model_file_name=self.model_name+"_"+suffix+".pth"
        save_path=os.path.join(self.save_path,model_file_name)
        torch.save(checkpoint,save_path)
        print("save checkpoint at {}".format(save_path))
        self.last_savepath=save_path
    def train(self,epoch):
        self.current_epoch=epoch
        criterion = nn.CrossEntropyLoss()
        total_losses=0
        start = time.strftime("%H:%M:%S")
        print("Starting epoch: %d | phase: train | ⏰: %s " % (epoch, start))
        self.train_bar=tqdm(self.train_data,desc='\r')
        for i, (data,label) in enumerate(self.train_bar):
            features={}
            for key, value in data.items():
                features[key]=value.clone().detach().to(self.device)
            output=self.model(features=features)
            label=label.reshape([label.shape[0]])
            loss=criterion(output,label.to(self.device))
            self.optimizer.zero_grad()
            total_losses+=float(loss)
            loss.backward()
            self.optimizer.step()
        print("Train loss: %.5f" % (total_losses / (i + 1)),'loss')
    def valid(self, epoch):
        self.model.eval()
        start = time.strftime("%H:%M:%S")
        criterion=nn.CrossEntropyLoss()
        total_losses=0
        print("Starting epoch: %d | phase: valid | ⏰: %s " % (epoch, start))
        valid_bar=tqdm(self.val_data,desc='\r')
        for i,(data,label) in enumerate(valid_bar):
            with torch.no_grad():
                features={}
                for key,value in data.items():
                    features[key]=value.clone().detach().to(self.device)
                output=self.model(features=features)
                label=label.reshape([label.shape[0]])
                loss=criterion(output,label.to(self.device))
                total_losses+=float(loss)
        num_batch=len(self.val_data)
        if total_losses/num_batch < self.best_loss:
            self.best_loss=total_losses/num_batch
            self._save_checkpoint(epoch,False,'bestloss')