import os
from data_preprocessor import DataPreprocessor, CompositeDataPreprocessor
from feature_engineering import EnrichDFDataPreprocessor
import lightgbm as lgb 
import xgboost as xgb 
import catboost as cbt 
import numpy as np
import joblib


model_dict = {
    'lgb': lgb.LGBMRegressor(objective='regression_l1', n_estimators=50),
    'xgb': xgb.XGBRegressor(tree_method='hist', objective='reg:absoluteerror', n_estimators=500, early_stopping_rounds = 100),
    'cbt': cbt.CatBoostRegressor(objective='MAE', iterations=50),
}

def create_train_XY(df_train):

    df_train_ = enrich_df_with_features(df_train)    
    features = [c for c in df_train_.columns if c not in nonfeatures]
    X = df_train_.dropna(subset=['target'])[features].values
    Y = df_train_.dropna(subset=['target'])['target'].values
    return X, Y


def train_with_model(df_train, model_dict, modelname, n_fold, fold):

    X, Y = create_train_XY(df_train)
    model = model_dict[modelname]
    index = np.arange(len(X))
    match modelname:
        case 'lgb':
            model.fit(X[index%n_fold!=fold], Y[index%n_fold!=fold], 
                eval_set=[(X[index%n_fold==fold], Y[index%n_fold==fold])], 
                callbacks=[lgb.early_stopping(100)]
            )
        case 'xgb':
            model.fit(X[index%n_fold!=fold], Y[index%n_fold!=fold], 
                eval_set=[(X[index%n_fold==fold], Y[index%n_fold==fold])], 
            )
        case 'cbt':
            model.fit(X[index%n_fold!=fold], Y[index%n_fold!=fold], 
                eval_set=[(X[index%n_fold==fold], Y[index%n_fold==fold])], 
                early_stopping_rounds = 100
            )
    # models.append(model)
    joblib.dump(model, f'./models/{modelname}_{fold}.model')
    return model

def train(df_train, n_fold = 5):
    os.system('mkdir models')
    models = []
    for i in range(n_fold):
        print(f"Training fold {i}")
        models.append(train_with_model(df_train, model_dict, 'lgb', n_fold, i))
        # train(model_dict, 'xgb', i)
        # train(model_dict, 'cbt', i)
    
    return models