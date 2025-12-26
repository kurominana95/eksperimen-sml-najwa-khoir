from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from joblib import dump
import pandas as pd
import numpy as np
import os

def coerce_object_numeric_and_impute_zero(df, threshold=0.8):
    """
    - Mendeteksi kolom object yang mayoritas berisi angka
    - Mengubahnya ke numerik (coerce)
    - Nilai non-numerik / spasi -> 0
    """
    df = df.copy()
    for col in df.select_dtypes(include="object").columns:
        coerced = pd.to_numeric(
            df[col].astype(str).str.strip(),
            errors="coerce"
        )
        valid_ratio = coerced.notna().mean()
        if valid_ratio >= threshold:
            df[col] = coerced.fillna(0)
    return df

def preprocess_data(
    data,
    target_column,
    save_path,
    header_path,
    test_size=0.2,
    random_state=42,
    drop_columns=None
):
    # Drop kolom yang tidak relevan, misal customerID
    if drop_columns:
        data = data.drop(columns=drop_columns, errors="ignore")

    # Pisahkan fitur dan target
    X = data.drop(columns=[target_column])
    y = data[target_column].map({"Yes": 1, "No": 0})

    # Cleaning otomatis
    X = coerce_object_numeric_and_impute_zero(X)

    # Sinkronkan y
    y = y.loc[X.index]

    # Simpan header awal fitur
    pd.DataFrame(columns=X.columns).to_csv(header_path, index=False)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state
    )

    # Deteksi tipe fitur
    numeric_features = X_train.select_dtypes(
        include=["int64", "float64"]
    ).columns.tolist()
    categorical_features = X_train.select_dtypes(
        include=["object"]
    ).columns.tolist()

    # Pipeline preprocessing
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features)
        ]
    )

    # Fit & transform
    X_train_transformed = preprocessor.fit_transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)

    # Simpan pipeline
    dump(preprocessor, save_path)

    return X_train_transformed, X_test_transformed, y_train, y_test, preprocessor, numeric_features, categorical_features

if __name__ == "__main__":
    RAW_PATH = "telco_churn_raw/data_raw.csv"
    os.makedirs("preprocessing", exist_ok=True)

    data = pd.read_csv(RAW_PATH)

    # Preprocess data dan drop customerID
    X_train, X_test, y_train, y_test, preprocessor, numeric_features, categorical_features = preprocess_data(
        data=data,
        target_column="Churn",
        save_path="preprocessing/preprocessor.joblib",
        header_path="preprocessing/header.csv",
        drop_columns=["customerID"]
    )

    # ===========================
    # Simpan CSV setelah preprocessing
    # ===========================
    os.makedirs("preprocessing/telco_churn_preprocessing", exist_ok=True)

    X_train_dense = X_train.toarray() if hasattr(X_train, "toarray") else X_train
    X_test_dense = X_test.toarray() if hasattr(X_test, "toarray") else X_test

    # Buat nama kolom lengkap (numerik + hasil one-hot)
    numeric_cols = numeric_features
    cat_cols = preprocessor.named_transformers_['cat']['encoder'].get_feature_names_out(categorical_features)
    all_cols = np.concatenate([numeric_cols, cat_cols])

    pd.DataFrame(X_train_dense, columns=all_cols).to_csv(
        "preprocessing/telco_churn_preprocessing/X_train.csv", index=False
    )
    pd.DataFrame(X_test_dense, columns=all_cols).to_csv(
        "preprocessing/telco_churn_preprocessing/X_test.csv", index=False
    )
    y_train.to_csv("preprocessing/telco_churn_preprocessing/y_train.csv", index=False)
    y_test.to_csv("preprocessing/telco_churn_preprocessing/y_test.csv", index=False)
