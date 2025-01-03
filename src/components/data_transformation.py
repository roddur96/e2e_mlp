import os, sys , pandas as pd , numpy as np  , pickle
from src.exception import CustomException
from src.logger import logging
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from src.utils import save_object

@dataclass 
class DataTransformationConfig :
    preprocessor_obj_file_path =os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config =DataTransformationConfig()

    def get_data_transformer_object(self):
        '''
        This function is responsible for data transformation
        '''
        try :
            num_cols =['reading_score','writing_score']
            cat_cols =["gender","race_ethnicity","parental_level_of_education","lunch","test_preparation_course"]

            num_pip =Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler() )
                ])
            cat_pip =Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy='most_frequent')),
                    ("onehotencoder", OneHotEncoder()),
                    ("scaler", StandardScaler(with_mean=False))
                ]
            )
            logging.info("pipeline completed")
            preprocessor =ColumnTransformer(
                [("num_pipeline",num_pip,num_cols),
                 ("col_pipeline", cat_pip,cat_cols)
                ]

            )
            return preprocessor
        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self, train_path , test_path):

        try:
            train_df= pd.read_csv(train_path)
            test_df =pd.read_csv(test_path)
            logging.info('data load and preprocessing object')

            preprocessing_obj =self.get_data_transformer_object()
            target_column_name ="math_score"

            input_feature_train_df = train_df.drop(columns=[target_column_name])
            target_feature_train_df=train_df[target_column_name]



            input_feature_test_df=test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df=test_df[target_column_name]

            logging.info(f"Applying preprocessing object on training dataframe and testing dataframe." )

            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info(f"Saved preprocessing object.")

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )


        except Exception as e:   
            logging.info('execption at transform')
            raise CustomException(e,sys)



