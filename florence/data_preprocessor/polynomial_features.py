import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from data_preprocessor.data_preprocessor import DataPreprocessor

class PolynomialFeaturesPreProcessor(DataPreprocessor):
    # def __init__(self, feature_name):
    #     super().__init__()
    #     self.feature_name = feature_name

    def apply(self, df):
        poly = PolynomialFeatures(2, interaction_only=True)
        df = poly.fit_transform(df).astype("float32")
        return df
