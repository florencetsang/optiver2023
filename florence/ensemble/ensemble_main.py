from os import listdir
from os.path import isfile, join
import random
from keras.models import load_model
import joblib
from ensemble.util import train_ensemble_model, prep_data, eval_ensemble_model
from scikeras.wrappers import KerasRegressor


X_train_fold, y_train_fold, X_val_fold, y_val_fold = prep_data()
base_path_of_model_pool = './ensemble/'
pool_path = [f for f in listdir(base_path_of_model_pool) if str(f).endswith('pool')]
# model_name = [f for f in listdir(base_path_of_model_pool) if isfile(join(base_path_of_model_pool, f))]

trial = 1
ensemble_num = 3
train_history = []
best_score=-1
best_model=None

for i in range(trial):
    model_pool = [join(pool, random.sample(listdir(join(base_path_of_model_pool, pool)), k=1)[0]) for pool in pool_path]
    # model_pool = random.sample(model_name, k=ensemble_num)
    estimators = [(str(model_path), KerasRegressor(load_model(join(base_path_of_model_pool, model_path)))) if str(model_path).endswith('keras') else (str(model_path), joblib.load(join(base_path_of_model_pool, model_path))) for model_path in model_pool]
    stacking_regressor, regressor_score = train_ensemble_model(estimators, X_train_fold, y_train_fold, X_val_fold, y_val_fold)
    train_history.append(([str(model_path) for model_path in model_pool], regressor_score))
    if regressor_score > best_score:
        best_score = regressor_score
        best_model = (estimators, stacking_regressor)


eval_ensemble_model(best_model[0], best_model[1], X_val_fold, y_val_fold)
print(str(best_model[0]))
save_path = f"Best_ensemble_model/{str(best_model[0])}"
joblib.dump(best_model[1], save_path)
