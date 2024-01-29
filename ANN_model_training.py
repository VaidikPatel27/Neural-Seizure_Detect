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
import matplotlib.pyplot as plt

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
sc = StandardScaler()
X_train_trf = sc.fit_transform(X_train)
X_test_trf = sc.transform(X_test)


# model training

seizure_pred_train, seizure_pred_test = y_train.iloc[:,0] , y_test.iloc[:,0]
lpd_pred_train, lpd_pred_test = y_train.iloc[:,1] , y_test.iloc[:,1]
gpd_pred_train, gpd_pre_test = y_train.iloc[:,2] , y_test.iloc[:,2]
lrda_pred_train, lrda_pred_test = y_train.iloc[:,3] , y_test.iloc[:,3]
grda_pred_train, grda_pred_test = y_train.iloc[:,4] , y_test.iloc[:,4]
other_pred_train, other_pred_test = y_train.iloc[:,5] , y_test.iloc[:,5]

output_train = [seizure_pred_train, lpd_pred_train, gpd_pred_train,
                lrda_pred_train, grda_pred_train, other_pred_train]
output_test = [seizure_pred_test, lpd_pred_test, gpd_pre_test,
               lrda_pred_test, grda_pred_test, other_pred_test]

input_layer = Input(shape = (426,))

hidden_layer = Dense(100,activation = 'relu')(input_layer)
hidden_layer = Dense(30,activation = 'sigmoid')(hidden_layer)

seizure_output = Dense(seizure_pred_train.nunique(),
                       activation = 'softmax',
                       name = 'seizure_output')(hidden_layer)

lpd_output = Dense(lpd_pred_train.nunique(),
                       activation = 'softmax',
                       name = 'lpd_output')(hidden_layer)

gpd_output = Dense(gpd_pred_train.nunique(),
                       activation = 'softmax',
                       name = 'gpd_output')(hidden_layer)

lrda_output = Dense(lrda_pred_train.nunique(),
                       activation = 'softmax',
                       name = 'lrda_output')(hidden_layer)

grda_output = Dense(grda_pred_train.nunique(),
                       activation = 'softmax',
                       name = 'grda_output')(hidden_layer)

other_output = Dense(other_pred_train.nunique(),
                       activation = 'softmax',
                       name = 'other_output')(hidden_layer)

output_layer = [seizure_output, lpd_output, gpd_output, lrda_output, grda_output, other_output]


model = tf.keras.Model(inputs=input_layer,
                       outputs=output_layer)

losses = ['sparse_categorical_crossentropy', 'sparse_categorical_crossentropy', 'sparse_categorical_crossentropy',
          'sparse_categorical_crossentropy', 'sparse_categorical_crossentropy', 'sparse_categorical_crossentropy']


model.compile(loss=losses,
              optimizer = keras.optimizers.Adam(learning_rate=0.0001),
              metrics = ['accuracy'],
              )

tf.keras.utils.plot_model(model, show_shapes=True, show_layer_activations=True, show_layer_names=True)
plt.show()

history = model.fit(X_train_trf,output_train,
                    validation_data = (X_test_trf, output_test),
                    epochs = 250,
                    batch_size = 20)


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.show()

plt.plot(history.history['other_output_accuracy'])
plt.plot(history.history['val_other_output_accuracy'])
plt.show()

y_pred_prob = model.predict(X_test_trf)
