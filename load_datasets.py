import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler


def load_breast_cancer():
    breast_df = pd.read_csv("datasets/breast_cancer.csv")
    breast_df.replace("?", pd.NA, inplace=True)
    breast_df.dropna(inplace=True)

    breast_y = breast_df["Class"]
    breast_df.drop(columns=["Sample_code_number", "Class"], axis=1, inplace=True)

    breast_cat_cols = breast_df.columns.values
    breast_cont_cols = []

    breast_df[breast_cat_cols] = breast_df[breast_cat_cols].apply(LabelEncoder().fit_transform)

    return breast_df, breast_y, breast_cat_cols, breast_cont_cols

def load_soybean_disease():
    soybean_df = pd.read_csv("datasets/soybean_disease.csv")
    soybean_df.replace("?", pd.NA, inplace=True)
    soybean_df.dropna(inplace=True)
    
    soybean_y = soybean_df["class"]
    soybean_df.drop(columns=["class"], axis=1, inplace=True)

    soybean_cat_cols = soybean_df.columns.values
    soybean_cont_cols = []

    soybean_df[soybean_cat_cols] = soybean_df[soybean_cat_cols].apply(LabelEncoder().fit_transform)

    return soybean_df, soybean_y, soybean_cat_cols, soybean_cont_cols

def load_bank_marketing():
    banking_df = pd.read_csv("datasets/bank_marketing.csv", sep=";")

    banking_y = banking_df["y"]
    banking_y = LabelEncoder().fit_transform(banking_y)

    banking_cat_cols = ["job", "marital", "education", "default", "housing", "loan", "contact", "poutcome"]
    banking_cont_cols = ["balance", "duration", "campaign", "pdays", "previous"]

    banking_df.drop(columns=["age", "y", "day", "month"], axis=1, inplace=True)
    banking_df[banking_cat_cols] = banking_df[banking_cat_cols].apply(LabelEncoder().fit_transform)
    banking_df[banking_cont_cols] = StandardScaler().fit_transform(banking_df[banking_cont_cols])
   
    return banking_df, banking_y, banking_cat_cols, banking_cont_cols

def load_census_income():
    census_df = pd.read_csv("datasets/census_income.csv")
    census_df.replace("?", pd.NA, inplace=True)
    census_df.dropna(inplace=True)

    census_df.loc[(census_df["class"] == " <=50K.") | (census_df["class"] == " <=50K"), "class"] = 0
    census_df.loc[(census_df["class"] == " >50K.") | (census_df["class"] == " >50K"), "class"] = 1
    census_y = census_df["class"].to_numpy()

    census_cat_cols = ["workclass", "education", "marital-status", "occupation", "relationship", "race", "sex", "native-country"]
    census_cont_cols = ["age", "fnlwgt", "education-num", "capital-gain", "capital-loss", "hours-per-week"]

    census_df.drop(columns="class", inplace=True)
    census_df[census_cat_cols] = census_df[census_cat_cols].apply(LabelEncoder().fit_transform)
    census_df[census_cont_cols] = StandardScaler().fit_transform(census_df[census_cont_cols])
    
    return census_df, census_y, census_cat_cols, census_cont_cols

def load_credit_approval():
    credit_df = pd.read_csv("datasets/credit_approval.csv")
    credit_df.replace("?", pd.NA, inplace=True)
    credit_df.dropna(inplace=True)
    
    credit_y = credit_df["A16"]
    credit_y = LabelEncoder().fit_transform(credit_y)

    credit_cat_cols = ["A1", "A4", "A5", "A6", "A7", "A9", "A10", "A12", "A13"]
    credit_cont_cols = ["A2", "A3", "A8", "A11", "A14", "A15"]

    credit_df.drop(columns="A16", inplace=True)
    credit_df[credit_cat_cols] = credit_df[credit_cat_cols].apply(LabelEncoder().fit_transform)
    credit_df[credit_cont_cols] = StandardScaler().fit_transform(credit_df[credit_cont_cols])
        
    return credit_df, credit_y, credit_cat_cols, credit_cont_cols

def load_heart_disease():
    heart_df = pd.read_csv("datasets/heart_disease.csv")
    heart_df.dropna(inplace=True)

    heart_y = heart_df["num"].to_numpy()

    heart_cat_cols = ["sex", "cp", "fbs", "restecg", "exang", "slope", "thal"]
    heart_cont_cols = ["age", "trestbps", "chol", "thalch", "oldpeak", "ca"]

    heart_df.drop(columns=["id", "dataset", "num"], inplace=True)
    heart_df.dropna(inplace=True)

    heart_df[heart_cat_cols] = heart_df[heart_cat_cols].apply(LabelEncoder().fit_transform)
    heart_df[heart_cont_cols] = StandardScaler().fit_transform(heart_df[heart_cont_cols])
    
    return heart_df, heart_y, heart_cat_cols, heart_cont_cols