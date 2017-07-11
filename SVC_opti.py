# -*- coding: utf-8 -*-
"""
Created on Sun Jul  2 18:41:18 2017

@author: Sergio
"""

#%%

from imbalance_funciones import *
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, make_scorer, f1_score
from skopt import gp_minimize

X_train, y_train, X_test, y_test = load_data()
SEED = 7

metodos = [f_NCRTomek, f_NCRTomekSmote, f_NearMiss, f_OneSidedSelection, f_SmoteEnn, f_SmoteTomek, f_RUSSmote, f_TomekRUS]

nombres = list(map(np.str, metodos))
for i in range(len(nombres)):
    nombres[i] = nombres[i][nombres[i].find("f_")::].split(" ")[0][2:] #Guarda los nombres en el vector nombres

print("Comenzando el estudio por procesos gausianos con %i metodos" % len(metodos))

lista_Parametros = []
lista_ROC = []
lista_std = []
lista_fmeasure = []
lista_gmean = []
SCORER = make_scorer(roc_auc_score)

for method, name in zip(metodos,nombres):
    
    
    print(name)
    X_train_p, y_train_p = method(X_train, y_train, SEED)
    print("Comenzando SVC")
    
    space = {
        'C': (0.1,5),
        'kernel': ['rbf','linear','poly','sigmoid'],
        'degree': [1,2,3,4,5,6,7],
        'gamma': (0.01,5)
        }

    svc = SVC()
    
    def objective(params):
        """ Wrap a cross validated inverted `accuracy` as objective func """
        svc.set_params(**{k: p for k, p in zip(space.keys(), params)})
        return 1-np.mean(cross_val_score(svc, X_train_p, y_train_p, cv=5, n_jobs=-1, scoring=SCORER))
    
    
    res_gp = gp_minimize(objective, space.values(), n_calls=50, random_state=SEED, verbose=False)
    best_hyper_params = {k: v for k, v in zip(space.keys(), res_gp.x)}
    
    print("Best ROC score =", 1-res_gp.fun)
    print("Best parameters =", best_hyper_params)
    
    
    params = best_hyper_params.copy()
    #params.update(params_fixed)
    
    clf = SVC(**params)
    clf.fit(X_train_p, y_train_p)
    y_test_preds = clf.predict(X_test)
    
    conf_mat = pd.crosstab(
        pd.Series(y_test, name='Actual'),
        pd.Series(y_test_preds, name='Predicted'),
        margins=True
    )
    
    lista_ROC.append(roc_auc_score(y_test, y_test_preds))
    lista_Parametros.append(best_hyper_params)
    lista_std.append(np.std(res_gp.func_vals))
    lista_fmeasure.append(f1_score(y_test, y_test_preds))
    lista_gmean.append(Gmean(y_test, y_test_preds))
    
    print('ROC: {0:.3f}'.format(roc_auc_score(y_test, y_test_preds)))
    
resultados = pd.DataFrame({"Metodos":nombres, "ROC":lista_ROC, "Parametros":lista_Parametros, "STD":lista_std, "Fmeasure":lista_fmeasure, "GMean":lista_gmean})
resultados.to_csv("SVC_resultados.csv")