from dataclasses import dataclass
from typing import List, Tuple
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import Dataset, DataLoader
import torch


class GenericDataset(Dataset):
    def __init__(self, df, cat_cols, cont_cols):
        self.cat = torch.tensor(df[cat_cols].values, dtype=torch.int)
        self.cat = torch.tensor(df[cont_cols].values, dtype=torch.float)
    
    def __getitem__(self, idx):
        return self.cat[idx], self.cont[idx]
    
    def __len__(self):
        return self.cat.shape[0]
    

class MixedTypeDataset:

    def __init__(self,     
        name: str,
        df: pd.DataFrame,
        y: np.array,
        cat_cols: List[str],
        cont_cols: List[str],
    ):
        self.name: str = name
        self.df: pd.DataFrame = df
        self.y: np.array = y
        self.cat_cols: List[str] = cat_cols
        self.cont_cols: List[str] = cont_cols

        dataset = GenericDataset(df, cat_cols, cont_cols)
        self.dataloader: DataLoader = DataLoader(dataset, batch_size=32)

        self.embedding_sizes: List[Tuple[int, int]] = [(df[col].nunique(), min(50, df[col].nunique()+1) // 2) for col in df[cat_cols]]
        self.cat_dim: int = sum(d for _, d in self.embedding_sizes)
        self.cont_dim: int = len(cont_cols)
        self.input_dim: int = self.cat_dim + self.cont_dim


def load_abalone():
    df = pd.read_csv("datasets/abalone.csv").sample(frac=1, random_state=0).reset_index(drop=True)
    df.dropna(inplace=True)

    y = df["Rings"]
    y = LabelEncoder().fit_transform(y)
    df.drop(columns=["Rings"], axis=1, inplace=True)

    cat_cols = ["Sex"]
    cont_cols = ["Length", "Diameter", "Height", "Whole_weight", "Shucked_weight", "Viscera_weight", "Shell_weight"]

    df[cat_cols] = df[cat_cols].apply(LabelEncoder().fit_transform)
    df[cont_cols] = StandardScaler().fit_transform(df[cont_cols])

    return MixedTypeDataset("Abalone", df, y, cat_cols, cont_cols)


def load_auction_verification():
    df = pd.read_csv("datasets/auction_verification.csv").sample(frac=1, random_state=0).reset_index(drop=True)
    df.dropna(inplace=True)

    y = df["verification.result"]
    y = LabelEncoder().fit_transform(y)
    df.drop(columns=["verification.result", "verification.time"], axis=1, inplace=True)

    cat_cols = ["process.b1.capacity", "process.b2.capacity", "process.b3.capacity", "process.b4.capacity", "property.product", "property.winner"]
    cont_cols = ["property.price"]

    df[cat_cols] = df[cat_cols].apply(LabelEncoder().fit_transform)
    df[cont_cols] = StandardScaler().fit_transform(df[cont_cols])

    df[cat_cols] = df[cat_cols].apply(LabelEncoder().fit_transform)
    df[cont_cols] = StandardScaler().fit_transform(df[cont_cols])

    return MixedTypeDataset("Auction Verification", df, y, cat_cols, cont_cols)


def load_bank_marketing(max_rows):
    df = pd.read_csv("datasets/bank_marketing.csv", sep=";").sample(frac=1, random_state=0).reset_index(drop=True)
    df.dropna(inplace=True)
    df = df.head(max_rows)

    y = df["y"]
    y = LabelEncoder().fit_transform(y)
    df.drop(columns=["age", "y", "day", "month"], axis=1, inplace=True)

    cat_cols = ["job", "marital", "education", "default", "housing", "loan", "contact", "poutcome"]
    cont_cols = ["balance", "duration", "campaign", "pdays", "previous"]

    df[cat_cols] = df[cat_cols].apply(LabelEncoder().fit_transform)
    df[cont_cols] = StandardScaler().fit_transform(df[cont_cols])

    return MixedTypeDataset("Bank Marketing", df, y, cat_cols, cont_cols)


def load_breast_cancer():
    df = pd.read_csv("datasets/breast_cancer.csv").sample(frac=1, random_state=0).reset_index(drop=True)
    df.replace("?", pd.NA, inplace=True)
    df.dropna(inplace=True)

    y = df["Class"]
    y = LabelEncoder().fit_transform(y)
    df.drop(columns=["Sample_code_number", "Class"], axis=1, inplace=True)

    cat_cols = df.columns.values.tolist()
    cont_cols = []

    df[cat_cols] = df[cat_cols].apply(LabelEncoder().fit_transform)

    return MixedTypeDataset("Breast Cancer", df, y, cat_cols, cont_cols)


def load_census_income(max_rows):
    df = pd.read_csv("datasets/census_income.csv").sample(frac=1, random_state=0).reset_index(drop=True)
    df.replace("?", pd.NA, inplace=True)
    df.dropna(inplace=True)
    df = df.head(max_rows)

    df.loc[(df["class"] == " <=50K.") | (df["class"] == " <=50K"), "class"] = 0
    df.loc[(df["class"] == " >50K.") | (df["class"] == " >50K"), "class"] = 1
    y = df["class"].to_numpy()
    df.drop(columns=["class"], axis=1, inplace=True)

    cat_cols = ["workclass", "education", "marital-status", "occupation", "relationship", "race", "sex", "native-country"]
    cont_cols = ["age", "fnlwgt", "education-num", "capital-gain", "capital-loss", "hours-per-week"]

    df[cat_cols] = df[cat_cols].apply(LabelEncoder().fit_transform)
    df[cont_cols] = StandardScaler().fit_transform(df[cont_cols])
    
    return MixedTypeDataset("Census Income", df, y, cat_cols, cont_cols)


def load_credit_approval():
    df = pd.read_csv("datasets/credit_approval.csv").sample(frac=1, random_state=0).reset_index(drop=True)
    df.replace("?", pd.NA, inplace=True)
    df.dropna(inplace=True)

    y = df["A16"]
    y = LabelEncoder().fit_transform(y)
    df.drop(columns=["A16"], axis=1, inplace=True)

    cat_cols = ["A1", "A4", "A5", "A6", "A7", "A9", "A10", "A12", "A13"]
    cont_cols = ["A2", "A3", "A8", "A11", "A14", "A15"]

    df[cat_cols] = df[cat_cols].apply(LabelEncoder().fit_transform)
    df[cont_cols] = StandardScaler().fit_transform(df[cont_cols])
        
    return MixedTypeDataset("Credit Approval", df, y, cat_cols, cont_cols)

def load_heart_disease():
    df = pd.read_csv("datasets/heart_disease.csv").sample(frac=1, random_state=0).reset_index(drop=True)
    df.dropna(inplace=True)

    y = df["num"]
    y = LabelEncoder().fit_transform(y)
    df.drop(columns=["id", "dataset", "num"], axis=1, inplace=True)

    cat_cols = ["sex", "cp", "fbs", "restecg", "exang", "slope", "thal"]
    cont_cols = ["age", "trestbps", "chol", "thalch", "oldpeak", "ca"]

    df[cat_cols] = df[cat_cols].apply(LabelEncoder().fit_transform)
    df[cont_cols] = StandardScaler().fit_transform(df[cont_cols])
    
    return MixedTypeDataset("Heart Disease", df, y, cat_cols, cont_cols)


def load_soybean_disease():
    df = pd.read_csv("datasets/soybean_disease.csv").sample(frac=1, random_state=0).reset_index(drop=True)
    df.replace("?", pd.NA, inplace=True)
    df.dropna(inplace=True)

    y = df["class"]
    y = LabelEncoder().fit_transform(y)
    df.drop(columns=["class"], axis=1, inplace=True)

    cat_cols = df.columns.values.tolist()
    cont_cols = []

    df[cat_cols] = df[cat_cols].apply(LabelEncoder().fit_transform)

    return MixedTypeDataset("Soybean Disease", df, y, cat_cols, cont_cols)


def load_all_datasets(max_rows) -> List[MixedTypeDataset]:
    return [
        load_abalone(),
        load_auction_verification(),
        load_bank_marketing(max_rows),
        load_breast_cancer(),
        load_census_income(max_rows),
        load_credit_approval(),
        load_heart_disease(),
        load_soybean_disease()
    ]
