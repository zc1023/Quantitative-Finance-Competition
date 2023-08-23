from lightgbm import LGBMRegressor 
import pandas as pd
from sklearn import linear_model
import pickle
import argparse

import wandb

# input argparser for hyperparameters
parser = argparse.ArgumentParser()
# add argument ratio defualt = 1
parser.add_argument('--boosting_type', type=str, default='gbdt')
parser.add_argument('--metric', type=str, default='rmse')
parser.add_argument('--num_leaves', type=int, default=31)
parser.add_argument('--feature_fraction', type=float, default=0.9)
parser.add_argument('--bagging_fraction', type=float, default=0.8)
parser.add_argument('--bagging_freq', type=int, default=5)
parser.add_argument('--learning_rate', type=float, default=0.05)
args = parser.parse_args()



train_df = pd.read_csv('./train.csv').set_index(['time_id', 'stock_id'])

# 填充缺失数据
train_df.fillna(0, inplace=True)
print(train_df)

reg = LGBMRegressor(boosting_type=args.boosting_type,  
                      objective='regression',  
                      metric=args.metric,  
                      num_leaves=args.num_leaves,  
                      learning_rate=0.05,  
                      feature_fraction=0.9,  
                      bagging_fraction=0.8,  
                      bagging_freq=5,  
                      verbose=0)  

# 将数据集划分为特征和目标变量
X = train_df.iloc[:,:-1].values
print(X)
y = train_df.iloc[:,-1].values

# 训练模型
reg.fit(X, y)
