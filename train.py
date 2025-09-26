from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
np.random.seed(0)
import scipy
from matplotlib import pyplot as plt
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
#from unet3d.scripts.train import main as get_data
project_dir = '/home/wanghao/wanghao/code'
#project_dir = '/media/xiao/新加卷/3DUnetCNN_local'
# tab_dir = os.path.join(project_dir,'table_data/clinic_data_cls_2.xlsx')
# train = pd.read_excel(tab_dir)
tab_dir = os.path.join(project_dir,'table_data/clinic_data.csv')
train = pd.read_csv(tab_dir)
target = 'label'

label0 = train[train["label"]==0]
label1 = train[train["label"]==1]
label2 = train[train["label"]==2]
if "Set" not in train.columns:
    label0["Set"] = np.random.choice(["train", "valid", "test"], p =[.8, .1, .1], size=(label0.shape[0],))
    label1["Set"] = np.random.choice(["train", "valid", "test"], p =[.8, .1, .1], size=(label1.shape[0],))
    label2["Set"] = np.random.choice(["train", "valid", "test"], p =[.8, .1, .1], size=(label2.shape[0],))

train_indices_0 = label0[label0.Set=="train"].index
valid_indices_0 = label0[label0.Set=="valid"].index
test_indices_0 = label0[label0.Set=="test"].index

train_indices_1 = label1[label1.Set=="train"].index
valid_indices_1 = label1[label1.Set=="valid"].index
test_indices_1 = label1[label1.Set=="test"].index

train_indices_2 = label2[label2.Set=="train"].index
valid_indices_2 = label2[label2.Set=="valid"].index
test_indices_2 = label2[label2.Set=="test"].index

train_indices = train_indices_0.append(train_indices_1).append(train_indices_2)
valid_indices = valid_indices_0.append(valid_indices_1).append(valid_indices_2)
test_indices = test_indices_0.append(test_indices_1).append(test_indices_2)


# if "Set" not in train.columns:
#     train["Set"] = np.random.choice(["train", "valid", "test"], p =[.8, .1, .1], size=(train.shape[0],))
#
# train_indices = train[train.Set=="train"].index
# valid_indices = train[train.Set=="valid"].index
# test_indices = train[train.Set=="test"].index

train = train.astype('str')
nunique = train.nunique()
types = train.dtypes

categorical_columns = ['']
categorical_dims =  {}
for col in train.columns:
    if types[col] == 'object':
        #print(col, train[col].nunique())
        l_enc = LabelEncoder()
        train[col] = train[col].fillna("VV_likely")
        train[col] = l_enc.fit_transform(train[col].values)
        categorical_columns.append(col)
        categorical_dims[col] = len(l_enc.classes_)
    else:
        train.fillna(train.loc[train_indices, col].mean(), inplace=True)




unused_feat = ['id_new','TSS', 'DIFF_ABS', 'NIHSS','SCORE','dice','gt','location','shifts','smokes','drinks','diabetes','myocardials','coronarys','atrias','hypertensions','strokes']
# unused_feat = ['id_new','dice','SCORE','M1','M2','M3','M4','M5','M6',
#                         'Caudate','Inular_ribbon','Lentiform_nucleus','Internal_capsule',
#                         'total', 'Caudate','Inular_ribbon',
#                         'Lentiform_nucleus','Internal_capsule', 'TSS', 'DIFF_ABS', 'NIHSS',
#                         'location', 'shifts', 'smokes', 'drinks', 'diabetes',
#                        'myocardials', 'coronarys', 'atrias', 'hypertensions', 'strokes']
unused_feat = ['id_new','TSS', 'DIFF_ABS', 'NIHSS','SCORE','dice','total',
               'location','shifts','smokes','drinks','diabetes','myocardials',
               'coronarys','atrias','hypertensions','strokes','Set']
features = [ col for col in train.columns if col not in unused_feat+[target]]

cat_idxs = [ i for i, f in enumerate(features) if f in categorical_columns]

cat_dims = [ categorical_dims[f] for i, f in enumerate(features) if f in categorical_columns]

#grouped_features = [[0, 1, 2], [8, 9, 10]]

tabnet_params = {
                    "cat_idxs":cat_idxs,
                    "cat_dims":cat_dims,
                    "cat_emb_dim":2,
                    "optimizer_fn":torch.optim.Adam,
                    "optimizer_params":dict(lr=2e-2,weight_decay=0.003),
                    "scheduler_params":{"step_size":30, # how to use learning rate scheduler
                                 "gamma":0.9},
                     "scheduler_fn":torch.optim.lr_scheduler.StepLR,
                     "mask_type":'entmax' # "sparsemax"
                     #"grouped_features" : grouped_features

                }

#train_loader,val_loader = get_data()
clf = TabNetClassifier(**tabnet_params
                      )
X_train = train[features].values[train_indices]
y_train = train[target].values[train_indices]

X_valid = train[features].values[valid_indices]
y_valid = train[target].values[valid_indices]

X_test = train[features].values[test_indices]
y_test = train[target].values[test_indices]
max_epochs = 100
from pytorch_tabnet.augmentations import ClassificationSMOTE
aug = ClassificationSMOTE(p=0.2)
#aug = None
# This illustrates the behaviour of the model's fit method using Compressed Sparse Row matrices
sparse_X_train = scipy.sparse.csr_matrix(X_train)  # Create a CSR matrix from X_train
sparse_X_valid = scipy.sparse.csr_matrix(X_valid)  # Create a CSR matrix from X_valid
from my_metric import my_metric
# Fitting the model
clf.fit(
    X_train=X_train, y_train=y_train,
    eval_set=[(X_train, y_train), (X_valid, y_valid)],
    eval_name=['train', 'valid'],
    #eval_name=['train'],
    #eval_metric=['accuracy'],
    eval_metric=['custom','accuracy','precision','recall','f1score'],
    max_epochs=max_epochs , patience=200,
    batch_size=256, virtual_batch_size=128,
    num_workers=0,
    weights=1,
    drop_last=False,
    augmentations=aug, #aug, None
)

# This illustrates the warm_start=False behaviour
plt.plot(clf.history['loss'])
plt.show()

# plot auc
plt.plot(clf.history['train_accuracy'])
plt.show()
plt.plot(clf.history['valid_accuracy'])
plt.show()

# plot learning rates
plt.plot(clf.history['lr'])
plt.show()


