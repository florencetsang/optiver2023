import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from data_preprocessor.data_preprocessor import DataPreprocessor

class PolynomialFeaturesPreProcessor(DataPreprocessor):
    def __init__(self, target_col_name='target'):
        super().__init__()
        self.target_col_name = target_col_name

    def apply(self, df):
        poly = PolynomialFeatures(2, interaction_only=True)
        features = [c for c in df.columns if c != self.target_col_name]
        x = df[features]
        y = df[self.target_col_name]
        
        x_polynomial_features = poly.fit_transform(x).astype("float32")
        new_features = poly.get_feature_names_out(x.columns)
        
        output_df = pd.DataFrame(x_polynomial_features, columns = new_features)  
        output_df.reset_index(drop=True, inplace=True)               
        y.reset_index(drop=True, inplace=True)
        # df_concat = pd.concat([output_df, y], axis=1) # full set
        df_concat = pd.concat([output_df.iloc[:, 0:int(len(new_features)/2)] , y], axis=1) # first half
        # df_concat = pd.concat([output_df.iloc[:, int(len(new_features)/2):len(new_features)] , y], axis=1) #second half
        return df_concat