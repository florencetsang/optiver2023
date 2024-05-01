from os import listdir
from os.path import isfile, join
import random
from keras.models import load_model
import joblib
from ensemble.util import (train_ensemble_model, prep_data, eval_ensemble_model, lgb_features, xgb_features, cat_features,
                           mlp_1_layer_features)
from scikeras.wrappers import KerasRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from utils.scoring_utils import ScoringUtils
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import make_pipeline
from data_preprocessor.normalization import NormalizationDataTransformer

import pandas as pd
import numpy as np

all_features = [('lgb_pool', lgb_features), ('xgb_pool', xgb_features), ('cb_pool', cat_features), ('mlp_1layer_pool', mlp_1_layer_features)]
# all_features = [('lgb_pool', lgb_features), ('mlp_1layer_pool', mlp_1_layer_features)]

X_train_fold, y_train_fold, X_val_fold, y_val_fold, df_train, df_val = prep_data()
base_path_of_model_pool = './ensemble/'
pool_path = [f for f in listdir(base_path_of_model_pool) if str(f).endswith('pool')]
# model_name = [f for f in listdir(base_path_of_model_pool) if isfile(join(base_path_of_model_pool, f))]

trial = 1
ensemble_num = 3
train_history = []
best_score=-1
best_model=None

for i in range(trial):
    model_pool = [join(model_features[0], random.sample(list(model_features[1]), k=1)[0]) for model_features in all_features]

    # model_pool = [join(pool, random.sample(listdir(join(base_path_of_model_pool, pool)), k=1)[0]) for pool in pool_path]
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
        elif str(estimator[0]).startswith('cb'):
            feature_dict = cat_features
        elif str(estimator[0]).startswith('mlp'):
            feature_dict = mlp_1_layer_features

        if str(estimator[0]).startswith('mlp'):
            estimator[1].initialize(pd.DataFrame(np.zeros((1, estimator[1].model.input_shape[1]))), pd.DataFrame(np.zeros((1))))
            col_transformer = ColumnTransformer([('select', 'passthrough', feature_dict[estimator[0].split('/')[-1]])],
                                                remainder='drop')
            col_transformer.set_output(transform='pandas')
            col_transformer.verbose_feature_names_out = False
            col_transformer.fit_transform(X_train_fold)
            last_fold_data_generator_transform_pipeline = make_pipeline(
                # FeatureToolsDFSTransformer(
                #     group_by_stock=True,
                #     group_by_date=False,
                #     group_by_stock_date=False,
                # ),
                # StockIdFeaturesDataTransformer(),
                NormalizationDataTransformer(
                    [
                        "imbalance_size",
                        "matched_size",
                        "bid_size",
                        "ask_size",
                    ],
                    "closing_movements",
                ),
                # RemoveIrrelevantFeaturesDataTransformer(['stock_id', 'date_id','time_id', 'row_id', "stock_date_id"]),
                verbose=True,
            )
            normalization_pipeline = last_fold_data_generator_transform_pipeline.fit(df_train)
            cur_pipeline = Pipeline([
                ('select', col_transformer),
                ('nomalization', normalization_pipeline),
                # remainder='drop' is the default, but I've included it for clarity
                (str(estimator[0]), estimator[1]),

            ])
            # cur_pipeline.fit_transform(X_train_fold)
            estimators_pipeline.append(
                (
                    str(estimator[0])
                    , cur_pipeline
                )
            )
        else:
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

for estimator_pipeline in best_model[1].estimators:
    estimator_score = ScoringUtils.calculate_mae([estimator_pipeline[1]], [df_val])
    print(f"model: {estimator_pipeline[0]}, MAE: {estimator_score}")
    pred = estimator_pipeline[1].predict(X_val_fold)
    r2 = r2_score(y_val_fold, pred)
    print(f"model: {estimator_pipeline[0]}, r2: {r2}")

stack_avg_mae = ScoringUtils.calculate_mae([best_model[1]], [df_val])
print(f"stack model: {estimator_pipeline[0]}, MAE: {stack_avg_mae}")
pred = best_model[1].predict(X_val_fold)
r2 = r2_score(y_val_fold, pred)
print(f"stack model: {estimator_pipeline[0]}, r2: {r2}")

# eval_ensemble_model(best_model[0], best_model[1], X_val_fold, y_val_fold)
print(str(best_model[0]))
save_path = f"ensemble/Best_ensemble_model/with_all_data_mlp_2"
joblib.dump(best_model[1], save_path)
