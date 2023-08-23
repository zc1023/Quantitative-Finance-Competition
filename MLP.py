# -*- coding: gbk -*-
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import pdb
import tqdm
from tqdm import tqdm
import pickle
#import matplotlib.pyplot as plt
import time
import argparse
import wandb


# input argparser for hyperparameters
parser = argparse.ArgumentParser()
# add argument ratio defualt = 1
parser.add_argument('--ratio', type=float, default=1)
parser.add_argument('--batch_size', type=int, default=1024*8)
parser.add_argument('--epochs', type=int, default=30)
parser.add_argument('--hidden_size', type=int, default=600)
parser.add_argument('--learning_rate', type=float, default=0.001)
args = parser.parse_args()

print("args.ratio",args.ratio)



# add super parameters into entity
wandb.init(project="finance-competition",config=args)
wandb.name = f"epoch_{args.epochs}__hiddensize_{args.hidden_size}__ratio-{args.ratio}"
#from torch.utils.tensorboard import SummaryWriter
#add time into log dir
# print current time
print(time.strftime('%m-%d_%H:%M',time.localtime(time.time())))
time_str = time.strftime('%m_%d_%H_%M',time.localtime(time.time()))




# Dataset
class baseDataset(Dataset):
    def __init__(self, dataframe,train_range = [0,300],label_range = 300):
        self.data = dataframe
        self.train_range = train_range
        self.label_range = label_range
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):

        features = self.data.iloc[idx,self.train_range[0]:self.train_range[1]].values
        label = self.data.iloc[idx,self.label_range]
        #print(features,label)

        features_tensor = torch.tensor(features, dtype=torch.float32)
        label_tensor = torch.tensor(label, dtype=torch.float32)
        
        return features_tensor, label_tensor
    
#data_process
train_df= pd.read_csv('train.csv').set_index(['time_id', 'stock_id'])
train_df.fillna(0, inplace=True)
print('==============dataset loaded')
print(train_df.head())


class baseModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(baseModel, self).__init__()
        self.norm = nn.BatchNorm1d(input_size)
        self.fc1 = nn.Linear(input_size, 150)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(150, 50)
        self.fc3 = nn.Linear(50, output_size)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = self.norm(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        #pdb.set_trace()
        x = 0.5*(self.sigmoid(x)-0.5)
        
        return x
    

dataset = baseDataset(train_df)


batch_size = args.batch_size
input_size = 300
hidden_size = args.hidden_size
output_size = 1



dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
model = baseModel(input_size, hidden_size, output_size).to("cuda")




num_epochs = 30
learning_rate = 1e-3
criterion = nn.MSELoss()
#optimizer = optim.(model.parameters(), lr=learning_rate)
optimizer = optim.AdamW(model.parameters(), lr=learning_rate,weight_decay = 1e-4,)

# writer = SummaryWriter(log_dir='runs/baseline_{}_epoch{}'.format(time_str,num_epochs))

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)


loss_list = []
data_iterator = tqdm(dataloader, total = len(dataloader)* num_epochs,mininterval=0.5) 

config = {"learning_rate": learning_rate, "batch_size": batch_size, "epochs": num_epochs, "hidden_size": hidden_size}
wandb.config.update(config)

for epoch in range(num_epochs):
    # use tqdm to show progress with dataloader
    for train_data, train_labels in data_iterator:
        train_data = train_data.to("cuda")
        outputs = model(train_data)
        train_labels = train_labels.to("cuda")
        loss = criterion(outputs, train_labels)
        #pdb.set_trace()
        # print sizes of outputs and labels
        #print(f"outputs:{outputs.size()}, labels:{train_labels.size()}")
        # print ouputs and labels in pairs
        #print(f"outputs:{outputs}, labels:{train_labels}")


        #print(f"loss:{loss}")
        loss_list.append(loss)
        # writer.add_scalar(tag="loss",  scalar_value=loss,    global_step= data_iterator.n)

        



        optimizer.zero_grad()
        loss.backward()
        params_to_clip = model.parameters()
        torch.nn.utils.clip_grad_norm_(params_to_clip, 1)
        optimizer.step()
        wandb.log({"train_loss": loss})


        # print loss for every 10 batches
        if data_iterator.n % 100 == 0:
            #print loss and ratio of the progress
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, {data_iterator.n/len(dataloader):.2%},{data_iterator.n}', end='\r')

        #print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

        # add ratio of data neede to trian 
        if data_iterator.n > args.ratio * len(dataloader):
            break
    scheduler.step()


# plt.plot(loss_list, label='loss')
    torch.save(model.state_dict(), 'basemodel_{}.pth'.format(time_str))

    test_df = pd.read_csv('./test.csv').set_index(['time_id', 'stock_id'])
    test_df.fillna(0, inplace=True)
    testset = baseDataset(test_df,label_range = 0)#nolabel
    batch_size = batch_size


    testdataloader = DataLoader(testset, batch_size=batch_size, shuffle=False)
    # X_test = torch.tensor(test_df.values)
    # X_test.size()

    test_df.head()

    print("len(test_df)/batch_size",len(test_df)/batch_size)
    print("len(testset)",len(testset))
    print("len(testdataloader)",len(testdataloader))

    model.eval()
    loss_list = []
    testdata_iterator = tqdm(testdataloader, total = len(testdataloader),mininterval=0.5)
    predict_labels = []

    for test_data, test_labels in testdata_iterator:
        test_data = test_data.to("cuda")
        test_labels = test_labels.to("cuda")
        # print sizes of inputs and labels
        # print(f"inputs:{test_data.size()}, labels:{test_labels.size()}")
        outputs = model(test_data)
        #loss = criterion(outputs, train_labels)
        predict_labels.append(outputs)

    #print(f"(predict_labels){(predict_labels)}")
    print(f"predict_labels[0].size(){predict_labels[1].size()}")
    predict_list = []

    # turn the tensors  in predict_labels into numpy arrays
    for i in range(len(predict_labels)):
        for j in predict_labels[i]:
            predict_list.append(j.detach().item())


    predict_list = predict_list[0:len(test_df)]
    len(predict_list)

    result = pd.DataFrame(predict_list, index = test_df.index, columns=['pred'])
    import os
    if not os.path.exists("results"):
        os.makedirs("results")
    result.to_csv(f'./results/basemodelresult{time_str}.csv')

    def rank_ic(result_path, label_path):
        test_label = pd.read_csv(label_path).set_index(['time_id', 'stock_id'])
        pred = pd.read_csv(result_path).set_index(['time_id', 'stock_id'])
        result = pd.concat([pred, test_label], axis=1)

        rank_ic_val = result.groupby('time_id').apply(lambda df: (df['pred'].rank()).corr(df['label'].rank())).mean()
        return rank_ic_val


    rank_ic_result = rank_ic(f'./results/basemodelresult{time_str}.csv',"./test_label.csv")

    print('rank_ic: ', rank_ic_result)#0.007,0.003

    # add rank_ic_result into wandb
    wandb.log({"rank_ic_result": rank_ic_result})
