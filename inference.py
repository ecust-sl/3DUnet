import numpy as np
from matplotlib import pyplot as plt
import scipy.stats as st
from sklearn import metrics
import pandas as pd
import os
from sklearn.metrics import roc_auc_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import math
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import cross_validate
from sklearn import svm
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score,precision_score, \
recall_score,f1_score,roc_auc_score,confusion_matrix
np.random.seed(0)
import scipy
from matplotlib import pyplot as plt
import os
import xgboost as xgb
import numpy as np
from sklearn.ensemble import RandomForestClassifier , GradientBoostingClassifier
from sklearn.model_selection import cross_validate
from sklearn.svm import SVC
import joblib
project_dir = '/home/shilei/project/code'
#project_dir = '/media/xiao/新加卷/3pythoDUnetCNN_local'
tab_dir = os.path.join(project_dir,'table_data/tabunet_uni_map-2.xlsx')
valid_data = os.path.join(project_dir, 'table_data/tabunet_uni_map_out.xlsx')
train = pd.read_excel(tab_dir)
valid_data = pd.read_excel(valid_data)
target = 'label'

unused_feat_1 = ['id' , 'NIHSS' , 'manual' , 'double' , 'uni_vol' , 'sum_map','score','shift']
unused_feat_2 = ['id' , 'NIHSS' , 'manual' , 'double' ,'score','shift']
features_1 = [col for col in train.columns if col not in unused_feat_1 + [target]]
features_2 = [col for col in train.columns if col not in unused_feat_2 + [target]]


fold = 5
skf = StratifiedKFold(n_splits=fold, shuffle=True, random_state=14)
X1 = train[features_1].values
X2 = train[features_2].values
X3 = valid_data[features_1].values
X4 = valid_data[features_2].values
y = train['label'].values
y_test = valid_data['label'].values

model_path = 'result_model12_pre/0323/random/model1/fold_3.pkl'

loaded_model = joblib.load(model_path)
pred_1 = loaded_model.predict(X3)
print(pred_1)
# pred_2 = loaded_model.predict(X4)

precision = precision_score(y_test, pred_1)

acc_score = accuracy_score(y_test, pred_1)
recall = recall_score(y_test, pred_1)
f1 = f1_score(y_test, pred_1)
auc_score = roc_auc_score(y_test, pred_1)
tn, fp, fn, tp = confusion_matrix(y_test, pred_1).ravel()
print(tn,  tp, fn, fp)
specificity = tn / (tn + fp + 1e-6)
sensity = tp / (tp + fn + 1e-6)
print('pre:', precision, 'acc:',acc_score, 'recall:' ,recall, 'f1:' ,f1, 'auc:',auc_score, 'spe:',specificity, 'sen:',sensity)
