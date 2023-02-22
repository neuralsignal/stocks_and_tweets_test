"""Naive Technical Analyst"""

import numpy as np

from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import accuracy_score, matthews_corrcoef
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression, RidgeCV

import pickle

from process_data import process_stocks
from score import scorer, processed_score


def main():
    # add average sentiment data or not
    save_name = 'fnew'
    add_sentiments = True
    Xtrain, Xval, Xtest, Ytrain, Yval, Ytest = process_stocks(
        add_sentiments=add_sentiments
    )
    
    # used for cross-validation
    X = np.vstack([Xtrain, Xval])
    Y = np.vstack([Ytrain, Yval])
    print(X.shape)
    pipe = Pipeline([
        ('scaler', StandardScaler()), 
        ('pca', PCA()), 
        ('model', MultiOutputRegressor(AdaBoostRegressor(), n_jobs=1))
    ])
    
    # gridsearch params
    n_comps = [25, 35, 45, 50, 55, 65, 75]
    n_ests = [100, 250]
    learning_rates = [0.005, 0.01, 0.02, 0.05, 0.1, 0.5]
    losses = ['linear', 'exponential', 'square']
    ests = [
        DecisionTreeRegressor(max_depth=3), 
        DecisionTreeRegressor(max_depth=3, criterion='absolute_error'),  
    ]
    
    pipe_params = [
        {
            'pca__n_components': n_comps, 
            'model': [MLPRegressor()], 
            'model__hidden_layer_sizes': [
                (1000, 100, 1000,), (50, 50, 50, 50,), 
                (100,), (50,)
            ], 
        },
        {
            'pca__n_components': n_comps, 
            'model': [RidgeCV(), LinearRegression()]
        },   
        {
            'pca__n_components': n_comps, 
            'model__estimator__estimator': [RidgeCV(), LinearRegression()], 
            'model__estimator__learning_rate': learning_rates, 
            'model__estimator__loss': losses, 
            'model__estimator__n_estimators': n_ests, 
        }, 
        {
            'pca__n_components': n_comps,
            'model__estimator__n_estimators': n_ests, 
            'model__estimator__estimator': ests, 
            'model__estimator__learning_rate': learning_rates, 
            'model__estimator__loss': losses, 
        },
    ]
    
    tscv = TimeSeriesSplit(n_splits=4, test_size=Yval.shape[0])
    
    gsearch = GridSearchCV(
        pipe, pipe_params, 
        cv=tscv, verbose=2, 
        n_jobs=-1, 
        scoring=scorer,
        
    )
    
    print('fitting')
    gsearch.fit(X, Y)
    print('done')
    
    add2path = ('sentiment_' if add_sentiments else '')
    with open(save_name + add2path + 'grid_model.pkl', 'wb') as f:
        pickle.dump(gsearch, f)
        
    print('predict')
    # test score for final model
    Ypred = gsearch.predict(Xtest)
    print('done')
    
    acc = np.array(
        [processed_score(accuracy_score, yt, ypred) for yt, ypred in zip(Ytest.T, Ypred.T)]
    )

    mcc = np.array(
        [processed_score(matthews_corrcoef, yt, ypred) for yt, ypred in zip(Ytest.T, Ypred.T)]
    )
    
    print(np.mean(acc))
    print(np.mean(mcc))
    
    np.save(save_name + add2path + 'accs.npy', acc)
    np.save(save_name + add2path + 'mccs.npy', mcc)
    
    print('the end')
    
    
if __name__ == '__main__':
    main()
