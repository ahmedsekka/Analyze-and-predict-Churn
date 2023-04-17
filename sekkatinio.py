#!/usr/bin/env python
# coding: utf-8

# In[33]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler, OrdinalEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold, learning_curve, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, roc_auc_score, roc_curve, make_scorer, recall_score
from xgboost import XGBClassifier
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from sklearn.inspection import permutation_importance
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')
import shap
shap.initjs()


# In[3]:


df = df = pd.read_csv("churn-bigml-80.csv")
dfm = df.copy()

le = LabelEncoder()
dfm['Churn'] = le.fit_transform(dfm['Churn'])
Xm = dfm.drop('Churn', axis=1)
ym = dfm['Churn']
Xm_train, Xm_test, ym_train, ym_test = train_test_split(Xm, ym, test_size=0.2, random_state=42)


# In[7]:


ratio = float(np.sum(ym_train == 0)) / np.sum(ym_train==1)
models = {
    'Logistic Regression': LogisticRegression(class_weight='balanced'),
    'KNN': KNeighborsClassifier(),
    'Random Forest': RandomForestClassifier(max_samples=0.5),
    'XGBoost': XGBClassifier(scale_pos_weight=ratio)
}
models1 = {
    'Logistic Regression': LogisticRegression(),
    'KNN': KNeighborsClassifier(),
    'Random Forest': RandomForestClassifier(),
    'XGBoost': XGBClassifier()
}


# In[8]:


categ_varsm = list(dfm.select_dtypes(include=['object', 'category']))
numeric_varsm = ['Account length',
 'Area code',
 'Number vmail messages',
 'Total day minutes',
 'Total day calls',
 'Total eve minutes',
 'Total eve calls',
 'Total night minutes',
 'Total night calls',
 'Total intl minutes',
 'Total intl calls',
 'Customer service calls']


# In[9]:


num_transformerm = Pipeline(steps=[('scaler', StandardScaler())])

ord_transformerm = Pipeline([
    ('ord', OrdinalEncoder())
])


preprocessorm = ColumnTransformer(
    transformers=[
        ('num', num_transformerm, numeric_varsm),
        ('cat', ord_transformerm, categ_varsm),
    ])


# In[10]:


def test_models(preprocessor, models):

    pipe = Pipeline([('preprocessor', preprocessor),
                    ('model', None)])
    
    for name, model in models.items():
        pipe.set_params(model=model)
        pipe.fit(Xm_train, ym_train)
        ym_pred_train = pipe.predict(Xm_train)
        ym_pred_test = pipe.predict(Xm_test)
        print(f'-------{name} Metrics:---------\n')
        print(f'Matrice de confusion Test: ')
        print(f'{confusion_matrix(ym_test, ym_pred_test)}\n')
        print(f'Classification report Test : ')
        print(f'{classification_report(ym_test, ym_pred_test)}')


# In[14]:


smote = SMOTE(random_state=42)

preprocessorm1 = ColumnTransformer(transformers=[
    ('num', num_transformerm, numeric_varsm),
    ('cat', ord_transformerm, categ_varsm)
])


# In[15]:


def test_smote(preprocessor, smote, models):

    pipe = Pipeline([('preprocessor', preprocessor),
                     ('smote', smote),
                     ('model', None)])
    
    for name, model in models.items():
        pipe.set_params(model=model)
        pipe.fit(Xm_train, ym_train)
        ym_pred_train = pipe.predict(Xm_train)
        ym_pred_test = pipe.predict(Xm_test)
        print(f'-------{name} Metrics:---------\n')
        print(f'Matrice de confusion Test: ')
        print(f'{confusion_matrix(ym_test, ym_pred_test)}\n')
        print(f'Classification report Test : ')
        print(f'{classification_report(ym_test, ym_pred_test)}')


# In[19]:


df1 = df.copy()
df2 = df.copy()


# In[20]:


df1['Churn'] = le.fit_transform(df1['Churn'])
df2['Churn'] = le.fit_transform(df1['Churn'])

df1.drop(['Total eve charge', 'Total day charge', 'Total night charge', 'Total intl charge'], axis=1, inplace=True)
df2.drop(['Total eve charge', 'Total day charge', 'Total night charge', 'Total intl charge'], axis=1, inplace=True)


# In[21]:


df2['Account length'] = pd.cut(df2['Account length'], bins=[0, 100, 125, df['Account length'].max()], labels=['[0:100]', '[100:125]', '[>125]'])
df2['Total day minutes'] = pd.cut(df2['Total day minutes'], bins=[-10 , 250, 400], labels=['[0:250]', '[>250]'])
df2['Total eve minutes'] = pd.cut(df2['Total eve minutes'], bins=[-10, 250, 450], labels=['[0:250]', '[>250]'])
df2['Total intl minutes'] = pd.cut(df2['Total intl minutes'], bins=[-5, 12, 30], labels=['[0:12]', '[>12]'])
df2['Total intl calls'] = pd.cut(df2['Total intl calls'], bins=[-1, 4, 30], labels=['[0:4]', '[>4]'])
df2['Customer service calls'] = pd.cut(df2['Customer service calls'], bins=[-2, 3, 12], labels=['[0:3]', '[>3]'])


# In[23]:


categ_vars1 = list(df1.select_dtypes(include=['object', 'category']))
numeric_vars1 = ['Account length',
 'Area code',
 'Number vmail messages',
 'Total day minutes',
 'Total day calls',
 'Total eve minutes',
 'Total eve calls',
 'Total night minutes',
 'Total night calls',
 'Total intl minutes',
 'Total intl calls',
 'Customer service calls']

categ_vars2 = list(df2.select_dtypes(include=['object', 'category']))
numeric_vars2 = ['Area code',
 'Number vmail messages',
 'Total day calls',
 'Total eve calls',
 'Total night minutes',
 'Total night calls']


# In[24]:


X = df1.drop('Churn', axis = 1)
y = df1['Churn']

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

data = {
    "X_train": X_train,
    "X_test": X_test,
    "y_train": y_train,
    "y_test": y_test
}


X1 = df2.drop('Churn', axis = 1)
y1 = df2['Churn']

X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size=0.2,stratify=y1, random_state=42)

data1 = {
    "X1_train": X1_train,
    "X1_test": X1_test,
    "y1_train": y1_train,
    "y1_test": y1_test
}

numerical_transformer = Pipeline(steps=[
    ('standard', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    #('ordinal', OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan))
    ('ordinal', OrdinalEncoder())
])

preprocessor = {
    'xgb': ColumnTransformer(transformers=[('num', numerical_transformer, numeric_vars1),('cat', categorical_transformer, categ_vars1)]),
    'lr':  ColumnTransformer(transformers=[('num', numerical_transformer, numeric_vars2),('cat', categorical_transformer, categ_vars2)])
}


# In[25]:


pipeline_xgb = Pipeline(steps=[
    ('preprocessor', preprocessor['xgb']),
    ('model', XGBClassifier(objective='binary:logistic'))
])

pipeline_xgb.fit(X_train, y_train)


# In[26]:


pipeline_lr = Pipeline(steps=[
    ('preprocessor', preprocessor['lr']),
    ('model', LogisticRegression(solver='saga' , penalty='l1', max_iter=1000, class_weight='balanced', C=1.2067926406393288))
])

pipeline_lr.fit(X1_train, y1_train)


# In[51]:


pipe = {
    'xgb': pipeline_xgb,
    'lr': pipeline_lr
}

clf = CalibratedClassifierCV(pipeline_xgb)
scorer = make_scorer(recall_score)
param_grid = {'clf__method': ['isotonic', 'sigmoid']}
kf = StratifiedKFold(n_splits=3, random_state=42, shuffle=True)
grid_search = GridSearchCV(
estimator=Pipeline([('clf', clf)]),
param_grid=param_grid,
cv=kf,
scoring = scorer
)
grid_search.fit(X_train, y_train)
        
best_method = grid_search.best_params_['clf__method']
    
final_pipeline = Pipeline([
    ('clf', CalibratedClassifierCV(pipe['xgb'], cv=kf, method=best_method))
                          ])
final_pipeline.fit(X_train, y_train)
y_cal = final_pipeline.predict(X_test)
y_pred = pipe['xgb'].predict(X_test)
    
y_cal_score = final_pipeline.predict_proba(X_test)[:, 1] 


# In[35]:


def calib_xgb(X_train, X_test, y_train, y_test, y_score, model_name):
    print("scores de base : ")
    print(f"{classification_report(y_test, y_pred)} \n")
    print("scores après calibration")
    print(f"{classification_report(y_test, y_cal)} \n")
    fraction_of_positives, mean_predicted_value = calibration_curve(y_test, y_score, n_bins=10)
    plt.plot(mean_predicted_value, fraction_of_positives, "s-", label=model_name)
    
    fraction_of_positives, mean_predicted_value = calibration_curve(y_test, y_cal_score, n_bins=10)
    plt.plot(mean_predicted_value, fraction_of_positives, 's-', label='%s' % best_method)
    
    plt.plot([0, 1], [0, 1], '--', color='gray', label='Référence')
    plt.xlabel('Probabilité moyenne prédite')
    plt.ylabel('Fraction de positifs')
    plt.legend()
    plt.show()


# In[36]:


y_len = np.zeros(len(y_test))


# In[37]:


def rec_prec_cal(y_test):
    print("---------- Precision-Recall graphe---------- \n")   
    precision, recall, threshold = precision_recall_curve(
    y_test, y_cal_score)
    tst_prt = pd.DataFrame({
    "threshold": threshold,
    "recall": recall[1:],
    "precision": precision[1:]
    })
    tst_prt_melted = pd.melt(tst_prt, id_vars = ["threshold"],value_vars = ["recall", "precision"])
    sns.lineplot(x = "threshold", y = "value",hue = "variable", data = tst_prt_melted)
    optimal_proba_cutoff = sorted(list(zip(np.abs(precision - recall), threshold)), key=lambda i: i[0], reverse=False)[0][1]
    print(f"threshold d'intersection est : {optimal_proba_cutoff:.2f} \n")
    print(f'scores de base : ')
    print(f'{classification_report(y_test, y_pred)}\n')
    print(f'scores calibrés après modification de threshold: ')
    print(f'{classification_report(y_test, y_len)}')


# In[38]:


def roc_auc(y_test, y_score, model_name):
    roc_auc = roc_auc_score(y_test, y_score)
    false_positive_rate, true_positive_rate, threshold = roc_curve(y_test, y_score)
    plt.figure(figsize=(20, 8))
    plt.plot(false_positive_rate, true_positive_rate, color='b', label='ROC curve (area = %0.3f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='r', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) for '+model_name)
    plt.legend(loc='lower right')
    plt.show()


# In[39]:


scoring = ['accuracy', 'precision', 'recall', 'f1']
kf = StratifiedKFold(n_splits=3,random_state=42,shuffle=True)
def lear_curve(X, y, scoring, pipe, model_name):
    fig, axs = plt.subplots(nrows=len(scoring)//2, ncols=2, figsize=(20, 8))
    axs = axs.flatten()
    for i, metric in enumerate(scoring):
        train_sizes, train_scores, test_scores = learning_curve(estimator=pipe,
                                                                X=X,
                                                                y=y,
                                                                train_sizes=np.linspace(0.1, 1.0, 5),
                                                                cv=kf,
                                                                scoring=metric)
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)

        axs[i].plot(train_sizes, train_scores_mean, label='train')
        axs[i].fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.2)
        axs[i].plot(train_sizes, test_scores_mean, label='test')
        axs[i].fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.2)
        axs[i].legend(loc='best')
        axs[i].set_title(metric+' '+model_name)
        axs[i].set_xlabel("Taille de l'échantillon d'entraînement")
        axs[i].set_ylabel("Score")

    plt.tight_layout()
    plt.show()


# In[40]:


def models_aff(pipe, data_dict, model_type):
    if model_type == 'xgb':
        X_train = data_dict['X_train']
        X_test = data_dict['X_test']
        y_train = data_dict['y_train']
        y_test = data_dict['y_test']
        
    
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        y_score = pipe.predict_proba(X_test)[:, 1]
        model_name = 'XGBoost'
        roc_auc(y_test, y_score, model_name)
        y_len[y_cal_score >= 0.39 ] = 1.
        calib_xgb(X_train, X_test, y_train, y_test, y_score, model_name)
        print('\n')
        print('threshold choisi : 0.39')
        rec_prec_cal(y_test)
        lear_curve(X, y, scoring, pipeline_xgb, model_name)
    
    elif model_type == 'lr':
        X_train = data_dict['X1_train']
        X_test = data_dict['X1_test']
        y_train = data_dict['y1_train']
        y_test = data_dict['y1_test']

        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        y_score = pipe.predict_proba(X_test)[:, 1]
        model_name = 'Logistic Regression'
        #rec_prec(y_test, y_score, y_pred)
        roc_auc(y_test, y_score, model_name) 
        lear_curve(X1, y1, scoring, pipeline_lr, model_name)
                
    else:
        print("Error: Model type not supported.")
        return  


# In[42]:


importance_lr = pipeline_lr['model']
importance_xgb = pipeline_xgb['model']


# In[43]:


def feature_importance():
    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(20, 12))
    #importance = pipeline_lr.steps[1][1].coef_[0]
    importance = pipeline_lr['model'].coef_[0]
    feat_importances = pd.Series(importance, index=X1_train.columns)
    top_feat_importances = feat_importances.nlargest(20)
    color_palette = sns.color_palette("Paired", len(top_feat_importances))
    top_feat_importances.plot(kind='barh', title='Feature Importance for Logistic Regression', color=color_palette)
    ax[1].set_xlabel('Importance')
    ax[1].invert_yaxis()
    ax[1].set_yticklabels(labels=top_feat_importances.index.tolist())

    feature_importances = importance_xgb.feature_importances_
    sorted_idx = feature_importances.argsort()
    sorted_features = [X_train.columns[i] for i in sorted_idx]
    ax[0].barh(sorted_features, feature_importances[sorted_idx], color=color_palette)
    ax[0].set_title('Feature Importance for XGBClassifier')
    ax[0].set_xlabel('Importance')

    plt.show()


# In[44]:


X_test_enc = preprocessor['xgb'].transform(X_test)
X1_test_enc = preprocessor['lr'].transform(X1_test)


# In[45]:


def permut_importance():
    fig, ax = plt.subplots(2, 1, figsize=(20,12))
    result = permutation_importance(importance_xgb, X_test_enc, y_test, n_repeats=10, random_state=42)
    importance = result.importances_mean
    sorted_idx = importance.argsort()
    features = X_test.columns
    color_palette = sns.color_palette("Paired", len(features))
    ax[0].barh(range(len(sorted_idx)), importance[sorted_idx], color=color_palette)
    ax[0].set_yticks(range(len(sorted_idx)), [features[i] for i in sorted_idx])
    ax[0].set_title("permutation importance for XGBoost")
    ax[0].set_xlabel('Permutation Importance')

    result1 = permutation_importance(importance_lr, X1_test_enc, y1_test, n_repeats=10, random_state=42)
    importance1 = result1.importances_mean
    sorted_idx1 = importance1.argsort()
    features1 = X1_test.columns
    ax[1].barh(range(len(sorted_idx1)), importance1[sorted_idx1], color=color_palette)
    ax[1].set_yticks(range(len(sorted_idx1)), [features1[i] for i in sorted_idx1])
    ax[1].set_title("permutation importance for Logistic regression")
    ax[1].set_xlabel('Permutation Importance')

    plt.show()


# In[46]:


X_train_transformed = preprocessor['xgb'].transform(X_train)
X_test_transformed = preprocessor['xgb'].transform(X_test)

X1_train_transformed = preprocessor['lr'].transform(X1_train)
X1_test_transformed = preprocessor['lr'].transform(X1_test)


explainer = shap.Explainer(pipeline_xgb['model'], X_train_transformed, feature_names=X_train.columns)

explainer1 = shap.Explainer(pipeline_lr['model'], X1_train_transformed, feature_names=X1_train.columns)

shap_values = {
    'xgb' : explainer(X_test_transformed),
    'lr' : explainer1(X1_test_transformed)
}


# In[47]:


def shap_bar():
    plt.figure()
    shap.plots.bar(shap_values['xgb'], max_display=20, show = False)
    plt.gcf().set_size_inches(20,12)
    plt.title("shap bar for XGBoost") 

    plt.figure()
    shap.plots.bar(shap_values['lr'], max_display=20, show = False)
    plt.gcf().set_size_inches(20,12)
    plt.title("Shap bar for Logistic regression")
    plt.show()


# In[48]:


def shap_beeswarm():
    plt.figure()
    shap.plots.beeswarm(shap_values['xgb'], max_display=20, show = False)
    plt.gcf().set_size_inches(15,11)
    plt.title("Shap beeswarm for XGBoost") 


    plt.figure()
    shap.plots.beeswarm(shap_values['lr'], max_display=20, show = False)
    plt.gcf().set_size_inches(15,11)
    plt.title("Shap beeswarm for Logistic regression")

    plt.show()


# In[49]:


feature = ['Customer service calls', 'Total eve calls', 'Total night minutes', 'Total eve minutes', 'Account length', 'Voice mail plan', 'Total night calls', 'Total intl minutes', 'Total intl calls', 'Total day calls', 'Total day minutes', 'Number vmail messages', 'State', 'Area code', 'International plan']


# In[50]:


def shap_scatter_lr(feature, shap_values):
    for i in range(0, len(feature)-1, 2):
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20,10))
        shap.plots.scatter(shap_values[:, feature[i]], ax=axes[0], show=False)
        shap.plots.scatter(shap_values[:, feature[i+1]], ax=axes[1], show=False)
        plt.show()

