# -*- coding: utf-8 -*-
Original file is located at
    https://colab.research.google.com/drive/1fOPtFl29_pqYi_Xq0IfcPfFHnh2CjTmM
#SecureSlice : Network Slicing analysis using Deep Learning
import numpy as np # linear algebra
import pandas as pd # data processing
train_dataset = pd.read_csv("/content/train_dataset.csv")
test_dataset = pd.read_csv("/content/test_dataset.csv")
print(f"train dataset shape : {train_dataset.shape}\n test dataset shape : {test_dataset.shape}")
train_dataset.head(10)
#Checking if NAN form of data is present in the dataset
print(train_dataset.isna().sum())
print(test_dataset.isna().sum())

"""Exploratory Data Analysis (EDA) for the dataset used for creating the various visulations for the EDA process here we have imported matplotlib and seaborn libraries of python."""
import matplotlib.pyplot as plt
import seaborn as sns
#'LTE/5g Category' probabilistic distributions.
fig = plt.figure(figsize = (32,15))
#For train dataset
plt.subplot(2,1,1)
plt.title("Train Dataset 'LTE/5g Category' Probabilistic Distribution with 'slice Type' ",fontsize =20)
train_lte_hist = sns.histplot(data = train_dataset,x = "LTE/5g Category",stat = "probability",
                              hue = "slice Type")
plt.xticks(train_dataset["LTE/5g Category"].value_counts().index,fontsize = 20)
plt.xlabel("LTE/5g Category",fontsize = 20)
plt.ylabel("Probability",fontsize = 20)
plt.yticks(np.arange(0.000,0.030,0.005),fontsize = 20);
plt.setp(train_lte_hist.get_legend().get_texts(),fontsize = '20')
plt.setp(train_lte_hist.get_legend().get_title(),fontsize = '20');
#For test dataset
plt.subplot(2,1,2)
plt.title("Test Dataset 'LTE/5g Category' Probabilistic Distribution",fontsize =20)
sns.histplot(data = test_dataset,x = "LTE/5g Category",stat = "probability")
plt.xticks(test_dataset["LTE/5g Category"].value_counts().index,fontsize = 20)
plt.xlabel("LTE/5g Category",fontsize = 20)
plt.ylabel("Probability",fontsize = 20)
plt.yticks(np.arange(0.00,0.06,0.01),fontsize = 20);
#Time feature with slice type.
plt.title("Train dataset 'Time' feature with 'slice Type' ")
time_slice_type = pd.concat([train_dataset["Time"],train_dataset["slice Type"]],axis = 1)
time = time_slice_type.value_counts().index.get_level_values(0)
slice_type = time_slice_type.value_counts().index.get_level_values(1)
sns.scatterplot(x = time,y = slice_type,hue = time_slice_type.value_counts().values)
print(train_dataset["Time"].describe())
plt.figure(figsize = (18,12))
sns.scatterplot(data = train_dataset,x = "Time", y= "LTE/5g Category",hue = "slice Type",
                palette = "deep")
plt.xlabel("LTE/5g Category",fontsize = 18)
plt.ylabel("Time",fontsize = 18)
#Packet loss rate
fig = plt.figure(figsize = (20,15))
#For train dataset
plt.subplot(2,1,1)
plt.title("Train Dataset 'Packet Loss Rate' Probabilistic Distribution ",fontsize =16)
train_lte_hist = sns.histplot(data = train_dataset,x = "Packet Loss Rate",stat = "probability")
plt.xticks(train_dataset["Packet Loss Rate"].value_counts().index,fontsize = 16)
plt.yticks(np.arange(0,0.5,0.1),fontsize = 16)
plt.xlabel("Packet Loss Rate",fontsize = 16)
plt.ylabel("Probability",fontsize = 16)
#For test dataset
plt.subplot(2,1,2)
plt.title("Test Dataset 'Packet Loss Rate' Probabilistic Distribution ",fontsize =16)
train_lte_hist = sns.histplot(data = test_dataset,x = "Packet Loss Rate",stat = "probability",
                             color = "red")
plt.xticks(test_dataset["Packet Loss Rate"].value_counts().index,fontsize = 16)
plt.yticks(np.arange(0,0.5,0.1),fontsize = 16)
plt.xlabel("Packet Loss Rate",fontsize = 16)
plt.ylabel("Probability",fontsize = 16)
plt.figure(figsize = (20,15))
#For train dataset
plt.subplot(2,1,1)
plt.title("Train Dataset 'Packet Loss Rate' line",fontsize = 18)
plt.plot(train_dataset["Packet Loss Rate"].value_counts(),marker = "o")
plt.xticks(train_dataset["Packet Loss Rate"].unique(),fontsize = 14)
plt.yticks(train_dataset["Packet Loss Rate"].value_counts(),fontsize = 14)
plt.xlabel("Packet Loss Rate",fontsize = 16)
plt.ylabel("Count",fontsize = 16)
#For test dataset
plt.subplot(2,1,2)
plt.title("Test Dataset 'Packet Loss Rate' line ",fontsize = 18)
plt.plot(test_dataset["Packet Loss Rate"].value_counts(),marker = "o",color = "green")
plt.xticks(test_dataset["Packet Loss Rate"].unique(),fontsize = 14)
plt.yticks(test_dataset["Packet Loss Rate"].value_counts(),fontsize = 14)
plt.xlabel("Packet Loss Rate",fontsize = 16)
plt.ylabel("Count",fontsize = 16)
plt.figure(figsize = (12,8))
#For train dataset
plt.subplot(1,2,1)
plt.title("Train Dataset 'Packet delay' counts with 'slice Type'")
sns.histplot(data = train_dataset,x = "Packet delay",hue = "slice Type",palette = "deep",
            kde = True)
#For test dataset
plt.subplot(1,2,2)
plt.title("Test Dataset 'Packet delay' probability distributions")
sns.histplot(data = train_dataset,x = "Packet delay",palette = "deep",stat = "probability",
            kde = True)
plt.figure(figsize = (12,6))
#For IoT
plt.subplot(1,2,1)
plt.title("Train Dataset 'IoT Devices' probability with slice Type")
sns.histplot(data = train_dataset,x = "slice Type",hue = "IoT Devices",
             palette = sns.color_palette("bright",3),stat = "probability",alpha = 0.5)
#For LTE/5G
plt.subplot(1,2,2)
plt.title("Train Dataset 'LTE/5G' probability with slice Type")
sns.histplot(data = train_dataset,x = "slice Type",hue = "LTE/5G",palette = "muted",stat = "probability"
            ,alpha = 0.5)
#Find the counts of each slice for GBR.
GBR_slice_type = pd.concat([train_dataset["GBR"],train_dataset["slice Type"]],axis = 1)
GBR = pd.DataFrame(GBR_slice_type.value_counts().index.get_level_values(0).values,columns = ["GBR"])
slice_type = pd.DataFrame(GBR_slice_type.value_counts().index.get_level_values(1).values,
                         columns = ["slice Type"])
counts = pd.DataFrame(GBR_slice_type.value_counts().values,columns = ["count"])
GBR_slice_type = pd.concat([GBR,slice_type,counts],axis = 1)
#Plot GBR with slice Type total counts.
print(GBR_slice_type)
sns.lmplot(data = GBR_slice_type, x = "GBR", y = "count",hue = "slice Type").set(
    title = "Train Dataset each slice type counts by 'GBR' feature")
sns.displot(data = train_dataset,x = "AR/VR/Gaming",y = "LTE/5G",hue = "slice Type",
            palette = "bright")
plt.figure(figsize = (14 ,8))
plt.subplot(1,2,1)
plt.title("Train Dataset 'slice Type' with respect to 'Healthcare' feature")
sns.histplot(data = train_dataset,x = "slice Type",hue = "Healthcare",stat = "probability",
            palette = "bright",alpha = 0.5)
plt.subplot(1,2,2)
plt.title("Train Dataset 'slice Type' with respect to 'Industry 4.0' feature")
sns.histplot(data = train_dataset,x = "slice Type",hue = "Industry 4.0",stat = "probability",
            palette = "flare",alpha = 0.5)
plt.figure(figsize = (16 ,6))
#Train Dataset 'Public Safety' and related 'Packet delay' with 'slice Type' hue.
plt.subplot(1,2,1)
plt.title("Train Dataset 'Public Safety' and related 'Packet delay' with 'slice Type' hue.")
sns.scatterplot(data = train_dataset,x = "Public Safety",y = "Packet delay",hue = "slice Type",
             palette = "bright",alpha = 0.5)
#Train Dataset 'Public Safety' and related 'GBR' with 'slice Type' hue.
plt.subplot(1,2,2)
plt.title("Train Dataset 'Public Safety' and related 'GBR' with 'slice Type' hue.")
sns.scatterplot(data = train_dataset,x = "Public Safety",y = "GBR",hue = "slice Type",
             palette = "bright",alpha = 0.5)
plt.figure(figsize = (16 ,6))
#Train Dataset 'Smart Transportation' and related 'GBR' with 'slice Type' hue.
plt.subplot(1,2,1)
plt.title("Train Dataset 'Smart Transportation' and related 'GBR' with 'slice Type' hue.")
sns.scatterplot(data = train_dataset,x = "Smart Transportation",y = "GBR",hue = "slice Type",
             palette = "bright",alpha = 0.5)
#Train Dataset 'Smart Transportation' and related 'Packet delay' with 'slice Type' hue.
plt.subplot(1,2,2)
plt.title("Train Dataset 'Smart Transportation' and related 'Packet delay' with 'slice Type' hue.")
sns.scatterplot(data = train_dataset,x = "Smart Transportation",y = "Packet delay",hue = "slice Type",
             palette = "bright",alpha = 0.5)
plt.figure(figsize = (14 ,8))
#For Train Dataset 'slice Type' with respect to 'Smartphone' feature.
plt.subplot(1,2,1)
plt.title("Train Dataset 'slice Type' with respect to 'Smartphone")
sns.histplot(data = train_dataset,x = "slice Type",hue = "Smartphone",stat = "probability",
            palette = "bright",alpha = 0.5)
#For Train Dataset 'Packet delay' with respect to 'Smartphone' feature.
plt.subplot(1,2,2)
plt.title("Train Dataset 'Packet delay' with respect to 'Smartphone' feature")
sns.histplot(data = train_dataset,x = "Packet delay",hue = "Smartphone",stat = "probability",
            palette = "flare",alpha = 0.5)
#Feature Extraction
corr_results = train_dataset.corr()
fig = plt.figure(figsize = (16,12))
sns.heatmap(corr_results,annot = True)
plt.show()
train_dataset.drop(columns = ["Time","IoT"],inplace = True)
test_dataset.drop(columns = ["Time","IoT"],inplace = True)
train_dataset.head(5)
test_dataset.head(5)
"""#Data Processing"""
import keras
from keras.utils import to_categorical
def label_processing(df):
    # No Standartization because the features are classification datas.
    target_y = df.iloc[:,-1:].values
    target_y = to_categorical(target_y)
    return target_y
# Get the processed y labels
train_y = label_processing(train_dataset)
print(train_dataset.iloc[:,-1:].values.shape)
print(train_y)
"""#Train and Validation Splitting"""
from sklearn.model_selection import train_test_split
X_train,X_val,Y_train,Y_val = train_test_split(train_dataset.iloc[:,:-1],train_y,random_state=43,test_size= 0.2)
print(f"{X_train.shape} , {X_val.shape}")
#Split test and val dataset.
X_val,X_test,Y_val,Y_test = train_test_split(X_val,Y_val,random_state = 43,test_size = 0.5)
print(f"{X_train.shape}, {X_test.shape}, {X_val.shape}")
"""#Deep Neural Networks"""
from keras.layers import Conv1D,ConvLSTM1D,Flatten,Dense,BatchNormalization,Dropout
from keras.models import Sequential
def build_model():
    model = Sequential()
    #Input layer
    model.add(Dense(8,activation = "relu",kernel_initializer = "normal",input_dim = 14))
    #Hidden layer 1
    model.add(Dense(16,activation = "relu",kernel_initializer = "normal"))
    #Dropout and Batch Normalization 1
    model.add(Dropout(0.2))
    model.add(BatchNormalization())
    #Hidden layer 2
    model.add(Dense(32,activation = "relu",kernel_initializer = "normal"))
    #Dropout and Batch Normalization 2
    model.add(Dropout(0.2))
    model.add(BatchNormalization())
    #Flatten
    model.add(Flatten())
    #Fully connected layer
    model.add(Dense(32,activation = "relu",kernel_initializer = "normal"))
    #Output layer
    model.add(Dense(4,activation = "softmax",kernel_initializer = "normal"))
    return model
model = build_model()
model.summary()
!pip install visualkeras
import visualkeras
from PIL import ImageFont
from collections import defaultdict
color_map = defaultdict(dict)
color_map[Conv1D]['fill'] = 'orange'
color_map[Dropout]['fill'] = 'black'
color_map[Dense]['fill'] = 'green'
color_map[Flatten]['fill'] = 'teal'
visualkeras.layered_view(model,legend = True,draw_volume = False,spacing = 20,
                        color_map = color_map)
"""#Compiling the model"""
import tensorflow as tf
model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = 0.001),
             loss = "categorical_crossentropy",metrics = ["accuracy"])
#Add dimension for convolution.
X_train = np.expand_dims(X_train,axis = -1)
X_val = np.expand_dims(X_val,axis = -1)
"""#Training and Evaluating the model"""
history = model.fit(X_train,Y_train,batch_size = 64,epochs = 20,
                         validation_data=(X_val,Y_val),
          callbacks = tf.keras.callbacks.EarlyStopping(monitor="val_loss",patience=3))
fig = plt.figure(figsize = (12,5))
epochs = len(history.history["accuracy"])
import matplotlib.pyplot as plt
plt.subplot(1,2,1)
plt.title("Train Accuracy and Val Accuracy")
plt.plot(range(epochs),history.history["accuracy"])
plt.plot(range(epochs),history.history["val_accuracy"])
plt.legend(["accuracy","val_accuracy"])
plt.subplot(1,2,2)
plt.title("Train Loss and Val Loss")
plt.plot(range(epochs),history.history["loss"])
plt.plot(range(epochs),history.history["val_loss"])
plt.legend(["loss","val_loss"])
"""#SecureSlice Model Prediction"""
#Evaluation of model on test dataset.
model.evaluate(X_test,Y_test)
#Prediction on X_test
preds_X_test = model.predict(X_test)
#Decode Y_test and predictions on X_test
Y_test = [np.argmax(Y_test[i]) for i in range(len(Y_test))]
preds_X_test_decoded = [np.argmax(preds_X_test[i]) for i in range(len(preds_X_test))]
from sklearn.metrics import classification_report
print(classification_report(Y_test,preds_X_test_decoded))
preds = model.predict(test_dataset)
print(preds)
#Decode predictions.
preds_decoded = [np.argmax(preds[i]) for i in range(len(preds))]
test_dataset["predicted_slice_type"] = preds_decoded
test_dataset
"""#XGBoost Model Prediction"""
!pip install xgboost
train_dataset.iloc[:,:-1]
from xgboost import XGBClassifier
xgb_model = XGBClassifier(learning_rate = 0.1,n_estimators = 10,objective = "multi:softmax",num_class = 3)
#Label encoder for training to start slices from 0.
from sklearn.preprocessing import LabelEncoder
LE = LabelEncoder()
xgb_y_train = LE.fit_transform(train_dataset["slice Type"])
#Training
xgb_model.fit(train_dataset.iloc[:,:-1],xgb_y_train)
#Prediction on x test
xgb_preds = xgb_model.predict(X_test)
xgb_preds = LE.inverse_transform(xgb_preds)
#Classification report
print(classification_report(Y_test,xgb_preds))
#Prediction on test dataset
xgb_preds_test = xgb_model.predict(test_dataset.iloc[:,:-1])
xgb_preds_test = LE.inverse_transform(xgb_preds_test)
print(xgb_preds_test)
"""#Comparison of results predicted by XGBoost Model and SecureSlice Model"""
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
# SecureSlice Model Results
secure_slice_preds_subset = preds_decoded[:3000]
Y_test_subset = Y_test[:3000]
# XGBoost Model Results
xgb_model_preds_subset = xgb_preds_test[:3000]
# Accuracy Comparison
secure_slice_accuracy_subset = accuracy_score(Y_test_subset, secure_slice_preds_subset)
xgb_model_accuracy_subset = accuracy_score(Y_test_subset, xgb_model_preds_subset)
# Precision, Recall, and F1 Score
secure_slice_precision = precision_score(Y_test_subset, secure_slice_preds_subset, average='weighted')
xgb_model_precision = precision_score(Y_test_subset, xgb_model_preds_subset, average='weighted')
secure_slice_recall = recall_score(Y_test_subset, secure_slice_preds_subset, average='weighted')
xgb_model_recall = recall_score(Y_test_subset, xgb_model_preds_subset, average='weighted')
secure_slice_f1 = f1_score(Y_test_subset, secure_slice_preds_subset, average='weighted')
xgb_model_f1 = f1_score(Y_test_subset, xgb_model_preds_subset, average='weighted')
print("SecureSlice Model Metrics (Subset):")
print(f"Accuracy: {secure_slice_accuracy_subset:.4f}")
print(f"Precision: {secure_slice_precision:.4f}")
print(f"Recall: {secure_slice_recall:.4f}")
print(f"F1 Score: {secure_slice_f1:.4f}")
print("\nXGBoost Model Metrics (Subset):")
print(f"Accuracy: {xgb_model_accuracy_subset:.4f}")
print(f"Precision: {xgb_model_precision:.4f}")
print(f"Recall: {xgb_model_recall:.4f}")
print(f"F1 Score: {xgb_model_f1:.4f}")
# Confusion Matrix Comparison
secure_slice_conf_matrix_subset = confusion_matrix(Y_test_subset, secure_slice_preds_subset)
xgb_model_conf_matrix_subset = confusion_matrix(Y_test_subset, xgb_model_preds_subset)
print("\nSecureSlice Model Confusion Matrix (Subset):")
print(secure_slice_conf_matrix_subset)
print("\nXGBoost Model Confusion Matrix (Subset):")
print(xgb_model_conf_matrix_subset)
# Classification Report
secure_slice_classification_report = classification_report(Y_test_subset, secure_slice_preds_subset)
xgb_model_classification_report = classification_report(Y_test_subset, xgb_model_preds_subset)
print("\nSecureSlice Model Classification Report (Subset):")
print(secure_slice_classification_report)
print("\nXGBoost Model Classification Report (Subset):")
print(xgb_model_classification_report)
# Comparing Accuracy in a Bar Chart
labels_subset = ["SecureSlice Model", "XGBoost Model"]
accuracies_subset = [secure_slice_accuracy_subset, xgb_model_accuracy_subset]
plt.bar(labels_subset, accuracies_subset, color=['blue', 'green'])
plt.title("Model Accuracy Comparison (Subset)")
plt.ylabel("Accuracy")
plt.show()
from sklearn.preprocessing import label_binarize
# Convert labels to one-hot encoding for multiclass classification
secure_slice_preds_one_hot = to_categorical(secure_slice_preds_subset)
xgb_model_preds_one_hot = to_categorical(xgb_model_preds_subset)
Y_test_one_hot = to_categorical(Y_test_subset)
# Precision-Recall Curve
secure_slice_precision, secure_slice_recall, _ = precision_recall_curve(Y_test_one_hot.ravel(), secure_slice_preds_one_hot.ravel())
xgb_model_precision, xgb_model_recall, _ = precision_recall_curve(Y_test_one_hot.ravel(), xgb_model_preds_one_hot.ravel())
plot_precision_recall_curve(secure_slice_precision, secure_slice_recall, "SecureSlice Model Precision-Recall Curve (Subset)")
plot_precision_recall_curve(xgb_model_precision, xgb_model_recall, "XGBoost Model Precision-Recall Curve (Subset)")
# ROC Curve
secure_slice_fpr, secure_slice_tpr, _ = roc_curve(Y_test_one_hot.ravel(), secure_slice_preds_one_hot.ravel())
xgb_model_fpr, xgb_model_tpr, _ = roc_curve(Y_test_one_hot.ravel(), xgb_model_preds_one_hot.ravel())
secure_slice_roc_auc = auc(secure_slice_fpr, secure_slice_tpr)
xgb_model_roc_auc = auc(xgb_model_fpr, xgb_model_tpr)
plot_roc_curve(secure_slice_fpr, secure_slice_tpr, secure_slice_roc_auc, "SecureSlice Model ROC Curve (Subset)")
plot_roc_curve(xgb_model_fpr, xgb_model_tpr, xgb_model_roc_auc, "XGBoost Model ROC Curve (Subset)")