import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import f1_score, accuracy_score
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.svm import SVC

path = f'path of getthing raining file in parquet format'
df_ = pd.read_parquet(path)

df_.drop(['eeg_sub_id','spectrogram_sub_id','label_id','expert_consensus'],axis=1,inplace=True)
df = df_.dropna(axis=0)

X = df.drop(['seizure_vote','lpd_vote', 'gpd_vote', 'lrda_vote', 'grda_vote', 'other_vote'],axis=1)
y_2 = df[['seizure_vote','lpd_vote', 'gpd_vote', 'lrda_vote', 'grda_vote', 'other_vote']]

y_2['seizure_vote'] = y_2['seizure_vote'].replace(19,17)

X_train, X_test, y_train, y_test = train_test_split(X,y_2,
                                                    test_size = 0.10,
                                                    shuffle = True,
                                                    random_state = 1)
# scaling data
sc = StandardScaler()
X_train_trf = sc.fit_transform(X_train)
X_test_trf = sc.transform(X_test)

model training

# lor
lor = LogisticRegression()

# sgd
sgd = SGDClassifier(max_iter = 100,
                    random_state = 1,
                    loss = 'log_loss',
                    penalty = 'elasticnet',
                    # alpha = 0.00001,
                    l1_ratio = 0.25,
                    n_jobs = -1,
                    )

# SVC
svc = SVC(probability = True,
          max_iter = 80,
          random_state = 1)

# Random Forest
rfc = RandomForestClassifier(n_estimators = 200)

# XGB
xgb = XGBClassifier(eta=0.5,
                    n_estimators=150,
                    objective='multi:softmax',
                    multi_strategy="one_output_per_tree",
                    max_delta_step = 3,                      # For imbalance classes
                    tree_method = 'hist'
                    )

estimators = [
    ('sgd',sgd),
    ('svc',svc),
    ('rf',rfc),
    ('xgb',xgb)
]

# Stacking
stk = StackingClassifier(estimators = estimators,
                         final_estimator = lor,
                         n_jobs = -1,
                         )
stk_multi = MultiOutputClassifier(stk, n_jobs=-1)


# This step can take more than 4-5 hours
stk_multi.fit(X_train_trf,y_train)

y_pred_stk = stk_multi.predict(X_test_trf)

# Model Eveluation
features = ['seizure_vote', 'lpd_vote', 'gpd_vote', 'lrda_vote', 'grda_vote', 'other_vote']
for n in range(0,6):
  vote = []
  for row in range(y_pred_stk.shape[0]):
    vote.append(y_pred_stk[row][n])
  print()
  print(features[n],'\n',round(f1_score(y_test.iloc[:,n],vote,average='weighted')*100,2),'%','\n',round(accuracy_score(y_test.iloc[:,n],vote)*100,2),'%')
