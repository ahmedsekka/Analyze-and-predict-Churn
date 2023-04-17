#!/usr/bin/env python
# coding: utf-8

# In[30]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import ipywidgets as widgets
from IPython.display import display


# In[24]:


df = pd.read_csv('churn-bigml-80.csv')
numeric_vars = df.select_dtypes(include=['int', 'float'])


# In[15]:


def plot_var(variable):
    plt.figure(figsize=(20,15))
    plt.plot(df[variable])
    plt.title(variable)
    plt.show()


# In[5]:


def hist_var(variable):
    plt.figure(figsize=(20,15))
    plt.hist(df[variable], bins=40)
    plt.title(variable)
    plt.show()


# In[6]:


def kde_var(variable):
    plt.figure(figsize=(15,10))
    sns.kdeplot(df[variable])
    plt.title(variable)
    plt.show()


# In[7]:


def kde_churn(variable):
    plt.figure(figsize=(20,15))
    sns.kdeplot(df[variable], hue=df['Churn'], common_norm=False)
    plt.title(variable)
    plt.show()


# In[18]:


def percent_churn():
    fix, axs = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))
    sns.countplot(x="Churn", data=df, ax=axs[0])
    # ajouter les compteurs de chaque bins
    for index, count in enumerate(df["Churn"].value_counts()):
        axs[0].text(index, count+30, count, ha="center")
    axs[0].set_title("histogramme")

    # % target plot pie
    grouped_churn = df['Churn'].value_counts()
    axs[1].pie(grouped_churn, labels=['Churn', 'Not Churn'], autopct="%1.2f%%")
    axs[1].set_title("Pourcentage")

    plt.show()


# In[20]:


def boxplot_var(column_name):
    plt.figure()
    plt.boxplot(df[column_name])
    plt.title(column_name)
    plt.show()


# In[21]:


def outliers_percent():
    outliers_percentages = []
    for col in numeric_vars:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        upper_limit = q3 + 1.5 * iqr
        lower_limit = q1 - 1.5 * iqr
        outliers = df[(df[col] < lower_limit) | (df[col] > upper_limit)][col]
        percent = round((len(outliers)/len(df[col])) * 100,2)
        outliers_percentages.append(percent)
        print(f"Pourcentage d'outliers pour {col} : {percent}%")

    num_var = numeric_vars.columns
    plt.figure(figsize=(20,10))
    plt.bar(num_var, outliers_percentages, color=['black', 'red', 'green', 'blue', 'cyan'])
    plt.xticks(rotation=45)
    plt.xlabel('Variables')
    plt.ylabel('Pourcentage d\'outliers')
    plt.title('Pourcentage d\'outliers pour chaque variable')
    plt.show()


# In[27]:


def plot_piechart(df, var, target, labels, title):
    fig, axs = plt.subplots(nrows=8, ncols=2, figsize=(8, 20))
    for i, ax in enumerate(axs.flatten()):
        grouped_dataset = df.groupby(target)[var[i]].sum()
        ax.pie(grouped_dataset, labels=labels, autopct='%1.2f%%')
        ax.set_title(f"{var[i]} by {target}", color="red")
    plt.show()


# In[28]:


def corr_lin():
    df_corr = df.corr()
    mask = np.triu(np.ones_like(df_corr,dtype=bool))
    plt.figure(figsize=(20,15))
    sns.heatmap(df_corr, mask=mask, annot=True)
    plt.xticks(rotation=45, ha='right')
    plt.show()


# In[32]:


def corr_nl():
    df_corr_nl = df.corr(method="spearman")
    mask = np.triu(np.ones_like(df_corr_nl, dtype=bool))
    plt.figure(figsize=(20,15))
    sns.heatmap(df_corr_nl, mask=mask, annot=True)
    plt.xticks(rotation=45, ha='right')
    plt.show()

