import numpy as np
import pandas as pd
from utils.preprocess_utils import extract_features

import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import multilabel_confusion_matrix

# Defining Multiclass log loss (# source kaggle)


def multiclass_log_loss(y_true, y_pred, eps=1e-15):
    """Generate multiclass log version loss metric

    Args:
        y_true ([list]): True value
        y_pred ([list]): Metrix with class prediction
                         One probability per class
        eps ([float], optional): Defaults to 1e-15.
    """
    if len(y_true.shape) == 1:
        y_true_ = np.zeros((y_true.shape[0],
                            y_pred.shape[1],
                            ))
        for i, val in enumerate(y_true):
            y_true_[i, val] = 1
        y_true = y_true_

    clip = np.clip(
        y_pred,
        eps,
        1-eps)

    rows = y_true.shape[0]
    vsota = np.sum(y_true * np.log(clip))

    return -1.0/rows * vsota


def adding_df(df_list):
    return pd.concat([i for i in df_list])


# Defining functions for creating all metrics
def metrics_result(y_true, y_pred,
                   algorithm_name):

    # Encode labels to {-1:0, 0:1, 1:2 }

    y_true = LabelEncoder().fit_transform(y_true)
    df_score = pd.DataFrame(
        columns=['balanced_loss',
                 'balanced_accuracy_score',
                 'precision',
                 'recall',
                 'f1_score'],
        index=[algorithm_name])
    b_loss = multiclass_log_loss(y_true, y_pred)
    if y_pred.shape[1] > 1:
        y_pred = np.argmax(y_pred, axis=1)
        #map_v = multilabel_confusion_matrix(y_true, y_pred)
        #print(map_v.shape)
        #sns.heatmap(np.transpose((map_v)), annot=True, cmap=sns.light_palette("Blue", as_cmap=True), linewidths=1)
        
        b_accuracy = balanced_accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted')
        recall = recall_score(y_true, y_pred, average='weighted')
        f1 = f1_score(y_true, y_pred, average='weighted')
    df_score.loc[algorithm_name, :] = b_loss, b_accuracy, precision, recall, f1
    return df_score

def kfold_metrics(X, y, skf, clf_logistic):
    """
    Create different metrics for kfold predictor
    """
    balanced_loss_list = []
    balanced_accuracy_list = []
    precision_list = []
    recall_list = []
    f1_score_list = []


    for train_idx, valid_idx in skf.split(X, y):
        x_train = X[train_idx]
        y_train = y[train_idx]


        x_valid = X[valid_idx]
        y_valid = y[valid_idx]


        feature_train, feature_valid, vocabulary = extract_features(
        x_train, x_valid)

        clf_logistic.fit(feature_train, y_train)
        predictions = clf_logistic.predict_proba(
        feature_valid) 

        res = metrics_result(y_valid, predictions, clf_logistic.__class__.__name__)
        ls, ac, prec, reca, f1 = res['balanced_loss'], res['balanced_accuracy_score'],res['precision'], res['recall'], res['f1_score']
        balanced_loss_list.append(ls)
        balanced_accuracy_list.append(ac)
        precision_list.append(prec)
        recall_list.append(reca)
        f1_score_list.append(f1)
 
    total_loss = np.mean(balanced_loss_list)
    total_accuracy = np.mean(balanced_accuracy_list)
    total_precision = np.mean(precision_list)
    total_recall = np.mean(recall_list)
    total_f1 = np.mean(f1_score_list)

    print(f'total loss = {total_loss}\n total_accuracy = {total_accuracy}\n total_precision = {total_precision}\n total_recall = {total_recall}\n total f1 score  = {total_f1}')

    df_score = pd.DataFrame(
        columns=['balanced_loss',
                 'balanced_accuracy_score',
                 'precision',
                 'recall',
                 'f1_score'],
        index=[clf_logistic.__class__.__name__])
    
    df_score.loc[clf_logistic.__class__.__name__, :] = total_loss, total_accuracy, total_precision, total_recall, total_f1
    

    return df_score
    