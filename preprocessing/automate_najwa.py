from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from joblib import dump
import pandas as pd
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

        # hanya kolom yang secara semantik numerik
        if valid_ratio >= threshold:
            df[col] = coerced.fillna(0)

    return df

def preprocess_data(
    data,
    target_column,
    save_path,
    header_path,
    test_size=0.2,
    random_state=42
):
    
    # 1. Pisahkan fitur dan target
    X = data.drop(columns=[target_column])
    y = y = data[target_column].map({"Yes": 1, "No": 0})

    # 2. Cleaning otomatis
    X = coerce_object_numeric_and_impute_zero(X)

    # sinkronkan y (index aman)
    y = y.loc[X.index]

    # 3. Simpan header fitur
    pd.DataFrame(columns=X.columns).to_csv(header_path, index=False)

    # 4. Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state
    )

    # 5. Deteksi tipe fitur
    numeric_features = X_train.select_dtypes(
        include=["int64", "float64"]
    ).columns.tolist()

    categorical_features = X_train.select_dtypes(
        include=["object"]
    ).columns.tolist()

    # 6. Pipeline preprocessing
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

    # 7. Fit & transform
    X_train = preprocessor.fit_transform(X_train)
    X_test = preprocessor.transform(X_test)

    # 8. Simpan pipeline
    dump(preprocessor, save_path)

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    import pandas as pd
    os.makedirs("preprocessing", exist_ok=True)
    os.makedirs("preprocessing/telco_churn_preprocessing", exist_ok=True)

    RAW_PATH = "telco_churn_raw/data_raw.csv"

    data = pd.read_csv(RAW_PATH)

    X_train, X_test, y_train, y_test = preprocess_data(
        data=data,
        target_column="Churn",
        save_path="preprocessing/preprocessor.joblib",
        header_path="preprocessing/header.csv"
    )

    # simpan dataset hasil preprocessing

    pd.DataFrame(X_train).to_csv("telco_churn_preprocessing/X_train.csv",index=False)
    pd.DataFrame(X_test).to_csv("telco_churn_preprocessing/X_test.csv",index=False)
    y_train.to_csv("telco_churn_preprocessing/y_train.csv",index=False)
    y_test.to_csv("telco_churn_preprocessing/y_test.csv",index=False)






