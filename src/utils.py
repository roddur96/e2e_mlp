import os , sys , pandas as pd , numpy as np , dill ,pickle
from src.exception import CustomException
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import r2_score


def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)


def evaluate_models(X_train , y_train,X_test,y_test,models, params, cv=3 ):
    try:
        report ={}
        for i in range(len(list(models))):
            model = list(models.values())[i]
            para = params[list(models.keys())[i]]

            rs = RandomizedSearchCV(model , para , n_jobs=3 , verbose=0 , cv=cv)
            rs.fit(X_train, y_train)

            model.set_params(**rs.best_params_)
            model.fit(X_train, y_train)
            y_pred =model.predict(X_test)
            score = r2_score(y_test, y_pred)
            report[list(models.keys())[i]]= score 
        return report
    except Exception as e:
        raise CustomException(e,sys)

    
