import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler


def load_breast_cancer():
    breast_df = pd.read_csv("datasets/breast_cancer.csv")
    breast_y = breast_df["Class"]
    breast_y.hist()
    breast_df.drop(columns=["Sample_code_number", "Class"], axis=1, inplace=True)

    breast_cat_cols = breast_df.columns.values
    breast_cont_cols = []

    breast_df[breast_cat_cols] = breast_df[breast_cat_cols].apply(LabelEncoder().fit_transform)
    return breast_df, breast_cat_cols, breast_cont_cols

def load_soybean_disease():
    soybean_df = pd.read_csv("datasets/soybean_disease.csv")
    soybean_df.drop_duplicates(inplace=True)
    soybean_y = soybean_df["class"]
    soybean_y.hist()
    soybean_df.drop(columns=["class"], axis=1, inplace=True)

    soybean_cat_cols = soybean_df.columns.values
    soybean_cont_cols = []

    soybean_df[soybean_cat_cols] = soybean_df[soybean_cat_cols].apply(LabelEncoder().fit_transform)
    return soybean_df, soybean_cat_cols, soybean_cont_cols