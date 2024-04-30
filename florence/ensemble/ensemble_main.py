from os import listdir
from os.path import isfile, join
import random
from keras.models import load_model
import joblib
from ensemble.util import train_ensemble_model, prep_data, eval_ensemble_model, lgb_features, xgb_features, cat_features
from scikeras.wrappers import KerasRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer


X_train_fold, y_train_fold, X_val_fold, y_val_fold = prep_data()
base_path_of_model_pool = './ensemble/'
pool_path = [f for f in listdir(base_path_of_model_pool) if str(f).endswith('pool')]
# model_name = [f for f in listdir(base_path_of_model_pool) if isfile(join(base_path_of_model_pool, f))]

trial = 10
ensemble_num = 3
train_history = []
best_score=-1
best_model=None

for i in range(trial):
    model_pool = [join(pool, random.sample(listdir(join(base_path_of_model_pool, pool)), k=1)[0]) for pool in pool_path]
    # model_pool = random.sample(model_name, k=ensemble_num)
    estimators = [(str(model_path), KerasRegressor(load_model(join(base_path_of_model_pool, model_path)))) if str(model_path).endswith('keras') else (str(model_path), joblib.load(join(base_path_of_model_pool, model_path))) for model_path in model_pool]
    estimators_pipeline = []
    for estimator in estimators:
        feature_dict = None
        if str(estimator[0]).startswith('lgb'):
            feature_dict = lgb_features
            estimator[1].device = 'cpu'
        elif str(estimator[0]).startswith('xgb'):
            feature_dict = xgb_features
        else:
            feature_dict = cat_features
        col_transformer = ColumnTransformer([('select', 'passthrough', feature_dict[estimator[0].split('/')[-1]])], remainder='drop')
        col_transformer.fit_transform(X_train_fold)
        cur_pipeline = Pipeline([
            ('select', col_transformer),
            # remainder='drop' is the default, but I've included it for clarity
            (str(estimator[0]), estimator[1])
        ])
        # cur_pipeline.fit_transform(X_train_fold)
        estimators_pipeline.append(
            (
                str(estimator[0])
                , cur_pipeline
            )
        )


    stacking_regressor, regressor_score = train_ensemble_model(estimators_pipeline, X_train_fold, y_train_fold, X_val_fold, y_val_fold)
    train_history.append(([str(model_path) for model_path in model_pool], regressor_score))
    if regressor_score > best_score:
        best_score = regressor_score
        best_model = (estimators, stacking_regressor)


eval_ensemble_model(best_model[0], best_model[1], X_val_fold, y_val_fold)
print(str(best_model[0]))
save_path = f"Best_ensemble_model/{str(best_model[0])}"
joblib.dump(best_model[1], save_path)
