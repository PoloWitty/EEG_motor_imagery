import numpy
import torch
from sklearn.base import BaseEstimator, ClassifierMixin
import torch.nn as nn
from torch.utils.data import DataLoader,TensorDataset
import wandb

class BrainNetv1_(nn.Module):
    def __init__(self,channels,dropout_p=0.25):
        '''
         param:
         channels: the channel num of the data
        '''
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(1,8,kernel_size=(channels,1),stride=1,padding=0),#(b,1,c,t) -> (b,8,1,t)
            nn.BatchNorm2d(8),
            nn.ELU(),
            nn.Dropout(dropout_p)
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(1,4,kernel_size=(1,2),stride=(1,2),padding=(0,0)),#(b,1,8,t) -> (b,4,8,t/2)
            nn.BatchNorm2d(4),
            nn.ELU(),
            nn.MaxPool2d((2,3)),#(b,4,8,t/2) -> (b,4,4,t/6)
            nn.Dropout(dropout_p),
        )

        self.block3 = nn.Sequential(
            nn.Conv2d(4,2,kernel_size=(2,5),stride=(2,5),padding=0),#(b,4,4,t/6) -> (b,2,2,t/30)
            nn.BatchNorm2d(2),
            nn.ELU(),
            nn.MaxPool2d((1,5)),# (b,2,2,t/30) -> (b,2,2,t/150)
            nn.Dropout(dropout_p)
        )
        self.classifier = nn.Sequential(
            nn.Linear(20,2),# 2*2*750/150=4*5=20
            nn.Softmax(dim=1)
        )

    def forward(self,x:torch.tensor):
        x = x.unsqueeze(1)# (b,1,c,t)
        x = self.block1(x)# (b,8,1,t)
        x = x.permute((0,2,1,3))# (b,1,8,t)
        x = self.block2(x)# (b,4,4,t/6)
        x = self.block3(x)# (b,2,2,t/150)
        x = x.flatten(start_dim=1)
        y = self.classifier(x)# (b,2)
        return y
    
class BrainNetv1(BaseEstimator,ClassifierMixin):
    def __init__(self,channels=13,epochs=10,batch_size=50,verbose=False,dropout_p=0.25,data='data'):
        self.model = BrainNetv1_(channels,dropout_p=dropout_p)
        self.channels = channels
        self.device ='cuda' if torch.cuda.is_available() else 'cpu'
        self.epochs = epochs
        self.batch_size = batch_size
        self.dropout_p = dropout_p
        self.data = data
        self.verbose = verbose
        self.FloatTensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor


    def fit(self,X:numpy.ndarray,y):
        if self.verbose:
            run = wandb.init(project='EEGnet-project',reinit=True)
            wandb.config.update({
                'epochs': self.epochs,
                'batch_size': self.batch_size,
                'dropout_p': self.dropout_p,
                'data': self.data
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
        optimizer = torch.optim.Adam(self.model.parameters())

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

    model = BrainNetv1(verbose=True,epochs=200,batch_size=100)
    cv = LeavePGroupsOut(n_groups=1)

    scores = cross_val_score(model, data_train, labels, groups=groups, cv=cv, n_jobs=1)

    # Printing the results
    class_balance = np.mean(labels == labels[0])
    class_balance = max(class_balance, 1. - class_balance)
    print("EEGnet Classification accuracy: %f(+/-%f) / Chance level: %f" % (scores.mean(),scores.std(),
                                                                class_balance))

