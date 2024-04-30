import joblib
from os.path import isfile, join
from ensemble.util import train_ensemble_model, prep_data, eval_ensemble_model, lgb_features, xgb_features, cat_features


for model_path, features in cat_features.items():
    base_path_of_model_pool = './ensemble/'
    if str(model_path).startswith('lgb'):
        path_prefix = 'lgb_pool'
    if str(model_path).startswith('xgb'):
        path_prefix = 'xgb_pool'
    else:
        path_prefix = 'cb_pool'
        cur_model = joblib.load(join(base_path_of_model_pool, path_prefix, model_path))
        feature_no = len(cur_model.feature_names_)
        print(f"{str(model_path)}: {feature_no}")