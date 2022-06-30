import numpy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader,TensorDataset
from sklearn.base import BaseEstimator, ClassifierMixin

from spikingjelly.clock_driven import neuron, surrogate, functional
import wandb

class BrainNetv3_(nn.Module):
    def __init__(self,T,tau:float,channels=13,v_threshold=1.0,v_reset=0.0):
        '''
         param:
         channels: the channel num of the data
        '''
        super().__init__()
        self.T = T
        self.tau = tau

        self.encoder_1 = nn.Sequential(
            nn.Conv2d(1,8,kernel_size=(channels,1),stride=1,padding=0),#(b,1,c,t) -> (b,8,1,t)
            neuron.LIFNode(v_threshold=v_threshold, v_reset=v_reset, surrogate_function=surrogate.ATan()),
            nn.MaxPool3d((2,1,5)),# (b,8,1,t) -> (b,4,1,t/5)
        )

        self.encoder_2 = nn.Sequential(
            nn.Conv2d(1,1,kernel_size=(3,3),padding=(1,1),stride=1),# (b,1,4,t/5) -> (b,1,4,t/5)
            neuron.LIFNode(v_threshold=v_threshold, v_reset=v_reset, surrogate_function=surrogate.ATan()),
            nn.MaxPool2d((2,5)),# (b,1,4,t/5) -> (b,1,2,t/25)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(60,30,bias=False),# 2*750/25=60
            neuron.LIFNode(tau=tau, v_threshold=v_threshold, v_reset=v_reset, surrogate_function=surrogate.ATan()),
            nn.Linear(30,2),
            neuron.LIFNode(tau=tau, v_threshold=v_threshold, v_reset=v_reset, surrogate_function=surrogate.ATan()),
        )

    def forward(self,x:torch.tensor):
        x = x.unsqueeze(1)# (b,c,t) -> (b,1,c,t)
        x = self.encoder_1(x)# 不参与时间累加的卷积
        x = x.permute(0,2,1,3)

        out_spikes_counter = self.classifier(self.encoder_2(x))
        for t in range(1, self.T):
            out_spikes_counter += self.classifier(self.encoder_2(x))

        return out_spikes_counter/self.T
    
class BrainNetv3(BaseEstimator,ClassifierMixin):
    def __init__(self,T=4,tau=2.0,channels=13,v_threshold=1.0,v_reset=0.0,epochs=10,batch_size=50,verbose=False,data='data'):
        self.model = BrainNetv3_(T,tau=tau,v_threshold=v_threshold,v_reset=v_reset)
        self.T = T
        self.tau = tau
        self.channels = channels
        self.v_threshold = v_threshold
        self.v_reset = v_reset

        self.device ='cuda' if torch.cuda.is_available() else 'cpu'
        self.epochs = epochs
        self.batch_size = batch_size
        self.data = data
        self.verbose = verbose
        self.FloatTensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor


    def fit(self,X:numpy.ndarray,y):
        if self.verbose:
            run = wandb.init(project='BrainNetv3-project',reinit=True)
            wandb.config.update({
                'epochs': self.epochs,
                'batch_size': self.batch_size,
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
                functional.reset_net(self.model)

                if self.verbose:
                    wandb.log({'loss':loss})
        
        if self.verbose:
            run.finish()
    
    def predict(self,X:numpy.ndarray):
        X = self.FloatTensor(X)

        self.model.eval()
        output = self.model(X)
        functional.reset_net(self.model)
        return torch.argmax(output,dim=-1).cpu()


if __name__=='__main__':
    import numpy as np
    from sklearn.model_selection import LeavePGroupsOut,cross_val_score

    data = np.load('data/train/data.npz')
    data_train = data['X']
    labels = data['y']
    groups = [1]*200+[2]*200+[3]*200+[4]*200

    model = BrainNetv3(verbose=False,epochs=300,batch_size=100)
    cv = LeavePGroupsOut(n_groups=1)

    scores = cross_val_score(model, data_train, labels, groups=groups, cv=cv, n_jobs=-1)

    # Printing the results
    class_balance = np.mean(labels == labels[0])
    class_balance = max(class_balance, 1. - class_balance)
    print("BrainNetv3 Classification accuracy: %f(+/-%f) / Chance level: %f" % (scores.mean(),scores.std(),
                                                                class_balance))

