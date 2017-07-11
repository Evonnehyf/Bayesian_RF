# -*- coding: utf-8 -*-
"""
Created on Tue Jun 27 01:22:28 2017

@author: Sergio

Functions combining several methods for an imbalanced data case

"""

import imblearn.under_sampling as us
import  imblearn.over_sampling as os
import imblearn.combine as co
import imblearn.ensemble as en
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def load_data(norm = True, zeroone = False):
    """
    X_train, y_train, X_test, y_test = load_data(norm = True)

    Devuelve los datos
    """
    lista = ["Edad.FI", "IMC.VM","TAS.VM","TAD.VM", "CT.VM", "HDL.VM", "LDL.VM"]
    lista_borrar = ["indice","obesity.cat","hypchol.cat"]
    dataset = pd.read_csv("./incliva/train.csv", delimiter=",")
    dataset = dataset.drop(lista_borrar, axis = 1)
    dataset_test = pd.read_csv("./incliva/test.csv", delimiter=",")
    dataset_test = dataset_test.drop(lista_borrar, axis = 1)

#    if norm:
#      sc = StandardScaler(with_mean = True)
#      X_train = sc.fit_transform(X_train)
#      X_test = sc.fit_transform(X_test)
      
    if norm:
      sc = StandardScaler(with_mean = True)
      dataset.loc[:,lista] = sc.fit_transform(dataset.loc[:,lista])
      dataset_test.loc[:,lista] = sc.fit_transform(dataset_test.loc[:,lista])
      
      
    X_train = dataset.iloc[:,:-1].as_matrix()
    y_train = dataset["mortality.cat"].as_matrix()
    X_test = dataset_test.iloc[:,:-1].as_matrix()
    y_test = dataset_test["mortality.cat"].as_matrix()
      
     
    
    if zeroone:
        print("Entre cero y uno")
        dataset.apply(lambda x: (x-min(x)) / (max(x) - min(x)))

    return X_train, y_train, X_test, y_test


def f_NCRTomek(X_train, y_train, seed):
    ncr = us.NeighbourhoodCleaningRule(n_neighbors = 50, n_jobs = -1)
    X_train, y_train = ncr.fit_sample(X_train, y_train)
    tlink = us.TomekLinks(random_state = seed)
    X_train, y_train = tlink.fit_sample(X_train, y_train)
    return(X_train, y_train)

def f_NCRTomekSmote(X_train, y_train, seed):
    print("NCR")
    ncr = us.NeighbourhoodCleaningRule(n_neighbors = 200, n_jobs = -1, random_state = seed)
    X_ncr, Y_ncr = ncr.fit_sample(X_train, y_train)
    print("TomekLink")
    tlink = us.TomekLinks(random_state = seed)
    X_tlink, y_tlink = tlink.fit_sample(X_ncr, Y_ncr)
    print("SMOTE")
    smote = os.SMOTE(k_neighbors = 5, ratio = 0.33)
    X_train, y_train = smote.fit_sample(X_tlink, y_tlink)
    return(X_train, y_train)

def f_TomekSmote(X_train, y_train, seed):
    tlink = us.TomekLinks()
    X_tlink, Y_tlink = tlink.fit_sample(X_train, y_train)
    smote = os.SMOTE(k_neighbors = 5)
    X_train, y_train = smote.fit_sample(X_tlink, Y_tlink)
    return(X_train, y_train)

def f_OneSidedSelection(X_train, y_train, seed):
    oss = us.OneSidedSelection(random_state = seed, n_neighbors=20)
    X_train, y_train = oss.fit_sample(X_train, y_train)
    return(X_train, y_train)


def f_NearMiss(X_train, y_train, seed):
    nm = us.NearMiss(version=3, return_indices=True, n_neighbors = 10, random_state = seed) 
    X_train, y_train, idx_res = nm.fit_sample(X_train, y_train)
    return(X_train, y_train)


def f_SmoteEnn(X_train, y_train, seed):
    Nrat = 4000 #Aproximadamente el numero de la clase minoritaria
    rat = 2 #Cuantas veces más de mayoritaria
    
    train = np.append(X_train, y_train[:, np.newaxis], axis = 1)
    vivos = train[train[:,-1] == 0]
    muertos = train[train[:,-1] == 1]
    
    nums1 = np.random.choice(len(vivos), size = Nrat*rat, replace = False)
    vivos1 = vivos[nums1[:Nrat]] #Estos voy a usar en el algoritmo
    vivos2 = vivos[nums1[Nrat:]]
    np.append(vivos1, muertos, axis = 0)[:,:-1].shape
    
    #Smote + ENN
    sm = co.SMOTEENN(random_state = seed)
    X_resampled, y_resampled = sm.fit_sample(np.append(vivos1, muertos, axis = 0)[:,:-1], np.append(vivos1, muertos, axis = 0)[:,-1]) 
    
    X_train = np.append(X_resampled, vivos2[:,:-1], axis = 0)
    y_train = np.append(y_resampled, vivos2[:,-1], axis = 0)
    
    #Barajamos los datos
    
    X_train = X_train[np.random.choice(len(X_train), len(X_train), replace = False)]
    y_train = y_train[np.random.choice(len(y_train), len(y_train), replace = False)] 
    return(X_train, y_train)

def f_SmoteTomek(X_train, y_train, seed):
    
    Nrat = 4000 #Aproximadamente el numero de la clase minoritaria
    rat = 2 #Cuantas veces más de mayoritaria


    train = np.append(X_train, y_train[:, np.newaxis], axis = 1)
    vivos = train[train[:,-1] == 0]
    muertos = train[train[:,-1] == 1]

    nums1 = np.random.choice(len(vivos), size = Nrat*rat, replace = False)
    vivos1 = vivos[nums1[:Nrat]] #Estos voy a usar en el algoritmo
    vivos2 = vivos[nums1[Nrat:]]
    np.append(vivos1, muertos, axis = 0)[:,:-1].shape

    #Smote + Tomek Links
    sm = co.SMOTETomek(random_state = seed)
    X_resampled, y_resampled = sm.fit_sample(np.append(vivos1, muertos, axis = 0)[:,:-1], np.append(vivos1, muertos, axis = 0)[:,-1]) 

    X_train = np.append(X_resampled, vivos2[:,:-1], axis = 0)
    y_train = np.append(y_resampled, vivos2[:,-1], axis = 0)

    #Barajamos los datos

    X_train = X_train[np.random.choice(len(X_train), len(X_train), replace = False)]
    y_train = y_train[np.random.choice(len(y_train), len(y_train), replace = False)] 
    return(X_train, y_train)
    
def f_CNN(X_train, y_train, seed):
    cnn = us.CondensedNearestNeighbour(random_state=seed)
    X_train, y_train = cnn.fit_sample(X_train, y_train)
    return(X_train, y_train)

def f_RUSSmote(X_train, y_train, seed):
    rus = us.RandomUnderSampler()
    X_rus, Y_rus = rus.fit_sample(X_train, y_train)
    smote = os.SMOTE(k_neighbors = 5)
    X_train, y_train = smote.fit_sample(X_rus, Y_rus)
    return(X_train, y_train)

def f_TomekRUS(X_train, y_train, seed):
    tomek = us.TomekLinks()
    X_tomek, Y_tomek = tomek.fit_sample(X_train, y_train)
    rus = us.RandomUnderSampler()
    X_train, y_train = rus.fit_sample(X_tomek, Y_tomek)
    return(X_train, y_train)

# Metricas

from sklearn.metrics import recall_score

def Gmean(y_true, y_pred):
    
    y_true_inv = np.abs(y_true - 1)
    y_pred_inv = np.abs(y_pred - 1)

    return np.sqrt(recall_score(y_true, y_pred) * recall_score(y_true_inv, y_pred_inv))
