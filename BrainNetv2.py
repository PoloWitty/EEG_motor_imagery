import numpy
import torch
from sklearn.base import BaseEstimator, ClassifierMixin
import torch.nn as nn
from torch.utils.data import DataLoader,TensorDataset
import wandb

class BrainNetv2_(nn.Module):
    def __init__(self,c,t,d_ratio:int,h,head,dropout_p=0.25):
        '''
         param:
         c: the channel num of the data
         t: the time num of the data
         d_ratio: down sample ratio (t/d_ratio)
         h: the num of channels after spatial filter layer
         head: num of heads in channel self-attention
         dropout_p: the probility in dropout
        '''
        super().__init__()
        d_m = int(t/d_ratio)
        assert d_m*d_ratio == t,'d_ratio:%i'%d_ratio

        # time down sample
        self.block1 = nn.Sequential(
            nn.Conv1d(1,1,kernel_size=1,padding=0),#(b*c,1,t) -> (b*c,1,t)
            nn.InstanceNorm1d(1),
            nn.ELU(),
            nn.AvgPool1d((d_ratio)),
            nn.Dropout(dropout_p)
        )

        # spatial filter
        self.block2 = nn.Sequential(
            nn.Conv1d(c,h,kernel_size=1,stride=1,padding=0),#(b,c,d_m) -> (b,h,d_m)
            nn.InstanceNorm1d(h),
            nn.ELU(),
            nn.Dropout(dropout_p),
        )

        # time attention
        self.time_attention = nn.Sequential(
            nn.Linear(d_m,d_m),
        )
        self.time_attention_ = nn.Sequential(
            nn.InstanceNorm1d(h),
            nn.ELU(),
            nn.Dropout(dropout_p)
        )

        # channel self attention
        self.channel_self_attention_Wq = nn.Linear(d_m,d_m,bias=False)
        self.channel_self_attention_Wk = nn.Linear(d_m,d_m,bias=False)
        self.channel_self_attention_Wv = nn.Linear(d_m,d_m,bias=False)
        self.channel_attention_norm = nn.Sequential(
            nn.InstanceNorm1d(h),
            nn.Dropout(dropout_p)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(h*d_m,2),
            nn.Softmax(dim=1)
        )

        self.c = c
        self.t = t
        self.d_m = d_m
        self.h = h
        self.head = head
        self.d_k = int(self.d_m/self.head)# because only use it in channel self attention layer
        assert self.d_k*self.head == self.d_m, 'd_m:%i,head:%i'%(d_m,head)


    def forward(self,x:torch.tensor):
        # time down sample
        x = x.flatten(end_dim=1).unsqueeze(1)# (b,c,t) -> (b*c,t) -> (b*c,1,t)
        x = self.block1(x)# (b*c,1,d_m)
        x = x.squeeze().reshape((-1,self.c,self.d_m))# (b,c,d_m)

        # spatial filter
        x = self.block2(x)# (b,h,d_m)

        # time attention
        x = x.reshape(-1,self.d_m)# (b*c,d_m)
        t_attention = self.time_attention(x)# (b*c,d_m)
        x = x * t_attention # (b*h,d_m)
        x = x.reshape(-1,self.h,self.d_m)# (b,h,d_m)
        x = self.time_attention_(x)# (b,h,d_m)
        
        # channel self-attention
        residual = x.clone()

        Q = self.channel_self_attention_Wq(x)# (b,h,d_m)
        Q = Q.reshape(-1,self.h,self.head,self.d_k)# (b,h,head,d_k)
        Q = Q.permute(0,2,1,3)# (b,head,h,d_k)
        Q = Q.reshape((-1,self.h,self.d_k))# (b*head,h,d_k)

        K = self.channel_self_attention_Wk(x)# (b,h,d_m)
        K = K.reshape((-1,self.h,self.head,self.d_k))# (b,h,head,d_k)
        K = K.permute(0,2,1,3)# (b,head,h,d_k)
        K = K.reshape((-1,self.h,self.d_k))# (b*head,h,d_k)
        K = K.permute(0,2,1)# (b*head,d_k,h) in case Q @ K

        V = self.channel_self_attention_Wv(x)# (b,h,d_m)
        V = V.reshape((-1,self.h,self.head,self.d_k))# (b,h,head,d_k)
        V = V.permute(0,2,1,3)# (b,head,h,d_k)
        V = V.reshape((-1,self.h,self.d_k))# (b*head,h,d_k)

        channel_attention = nn.functional.softmax(Q @ K,dim=-1)
        x = channel_attention @ V # (b*head,h,d_k)
        x = nn.functional.elu(x)# activate
        x = x.reshape((-1,self.head,self.h,self.d_k))# (b,head,h,d_k)
        x = x.permute(0,2,1,3)# (b,h,head,d_k)
        x = x.reshape(-1,self.h,self.d_m)# (b,h,d_m)

        x = x + residual
        x = self.channel_attention_norm(x)

        # classifier
        y = self.classifier(x)
        return y
    
class BrainNetv2(BaseEstimator,ClassifierMixin):
    def __init__(self,channels=13,t=750,h=6,d_ratio=10,head=5,epochs=10,batch_size=100,verbose=False,dropout_p=0.25,data='data'):
        '''
         param:
         channels: the channel num of the data
         t: the time num of the data
         h: the num of channels after spatial filter layer
         head: num of heads in channel self-attention
         dropout_p: the probility in dropout
         
         !!!: t要能被d_ratio整除，能被d_ratio*head整除
        '''
        self.model = BrainNetv2_(channels,t,d_ratio,h,head,dropout_p=dropout_p)
        self.channels = channels
        self.t = t
        self.d_ratio = d_ratio
        self.h = h
        self.head = head

        self.epochs = epochs
        self.batch_size = batch_size
        self.dropout_p = dropout_p

        self.device ='cuda' if torch.cuda.is_available() else 'cpu'
        self.data = data
        self.verbose = verbose
        self.FloatTensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor


    def fit(self,X:numpy.ndarray,y):
        if self.verbose:
            run = wandb.init(project='BrainNetv2',reinit=True)
            wandb.config.update({
                'epochs': self.epochs,
                'batch_size': self.batch_size,
                'dropout_p': self.dropout_p,
                'data': self.data,
                'd_ratio': self.d_ratio,
                'head': self.head
            })
        X = torch.FloatTensor(X)
        y = torch.tensor(y.astype(numpy.int64))# one hot 要求输入时LongTensor
        y = nn.functional.one_hot(y,num_classes=2)
        y = y.type(torch.float)
        dataloader = DataLoader(
            dataset=TensorDataset(X,y),
            batch_size=self.batch_size,
            shuffle=True
        )

        loss_fn = nn.BCELoss()
        optimizer = torch.optim.Adam(self.model.parameters(),lr=1e-4)

        self.model.train()
        self.model.to(self.device)
        for e in range(self.epochs):
            for data in dataloader:

                input,labels = data
                input = input.to(self.device)
                labels = labels.to(self.device)

                optimizer.zero_grad()
                output = self.model(input)
                loss = loss_fn(output,labels)

                loss.backward()
                optimizer.step()

                if self.verbose:
                    wandb.log({'loss':loss})
        
        if self.verbose:
            run.finish()
    
    def predict(self,X:numpy.ndarray):
        X = self.FloatTensor(X)

        self.model.eval()
        output = self.model(X)
        return torch.argmax(output,dim=-1).cpu()

if __name__=='__main__':
    import numpy as np
    from sklearn.model_selection import LeavePGroupsOut,cross_val_score

    data = np.load('data/train/data.npz')
    data_train = data['X']
    labels = data['y']
    groups = [1]*200+[2]*200+[3]*200+[4]*200

    model = BrainNetv2(verbose=True,epochs=1600,d_ratio=25,h=4)
    cv = LeavePGroupsOut(n_groups=1)

    scores = cross_val_score(model, data_train, labels, groups=groups, cv=cv, n_jobs=1)

    # Printing the results
    class_balance = np.mean(labels == labels[0])
    class_balance = max(class_balance, 1. - class_balance)
    print("BrainNetv2 Classification accuracy: %f(+/-%f) / Chance level: %f" % (scores.mean(),scores.std(),class_balance))