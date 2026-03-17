

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack


def preprocess(train, test):
    train["journal_text"] = train["journal_text"].fillna("")
    test["journal_text"] = test["journal_text"].fillna("")

    mean_sleep = train["sleep_hours"].mean()

    train["sleep_hours"] = train["sleep_hours"].fillna(mean_sleep)
    test["sleep_hours"] = test["sleep_hours"].fillna(mean_sleep)

    train["previous_day_mood"] = train["previous_day_mood"].fillna("unknown")
    test["previous_day_mood"] = test["previous_day_mood"].fillna("unknown")

    train["face_emotion_hint"] = train["face_emotion_hint"].fillna("unknown")
    test["face_emotion_hint"] = test["face_emotion_hint"].fillna("unknown")

    categorical_cols = ["ambience_type" ,"time_of_day" ,"previous_day_mood" ,"face_emotion_hint" ,"reflection_quality"]

    train_encoded = pd.get_dummies(train[categorical_cols])
    test_encoded = pd.get_dummies(test[categorical_cols])

    train_encoded, test_encoded = train_encoded.align(test_encoded, join='left', axis=1, fill_value=0)

    # Converting text into numerical data using TFIDF 

    tfidf = TfidfVectorizer(max_features=500)

    X_text_train = tfidf.fit_transform(train["journal_text"])
    X_text_test = tfidf.transform(test["journal_text"])

    # Combining numerical values

    numerical_cols = ["sleep_hours", "duration_min", "energy_level", "stress_level"]

    # Converting the dataframes above in numerical values 

    train[numerical_cols] = train[numerical_cols].apply(pd.to_numeric)
    test[numerical_cols] = test[numerical_cols].apply(pd.to_numeric)

    X_num_train = train[numerical_cols].astype(float).values
    X_num_test = test[numerical_cols].astype(float).values

    X_cat_train = train_encoded.astype(float).values
    X_cat_test = test_encoded.astype(float).values
    
    X_train = hstack([X_text_train, X_cat_train, X_num_train])
    X_test = hstack([X_text_test, X_cat_test, X_num_test])

    return X_train, X_test, tfidf, train_encoded.columns

def preprocess_single(df, tfidf, ref_columns, include_targets=True):

    # Fill missing values
    df["journal_text"] = df["journal_text"].fillna("")
    df["previous_day_mood"] = df["previous_day_mood"].fillna("unknown")
    df["face_emotion_hint"] = df["face_emotion_hint"].fillna("unknown")

   
    # Categorical encoding
    
    categorical_cols = [
        "ambience_type",
        "time_of_day",
        "previous_day_mood",
        "face_emotion_hint",
        "reflection_quality"
    ]

    df_encoded = pd.get_dummies(df[categorical_cols])

    df_encoded = df_encoded.reindex(columns=ref_columns, fill_value=0)


    X_text = tfidf.transform(df["journal_text"])

    numerical_cols = ["sleep_hours", "duration_min", "energy_level", "stress_level"]
    if include_targets:
        # normal case
        X_num = df[numerical_cols].astype(float).values
    else:
        # create placeholder values (only for stage 1)
        df["energy_level"] = 0
        df["stress_level"] = 0
        X_num = df[numerical_cols].astype(float).values

    X_cat = df_encoded.astype(float).values

    from scipy.sparse import hstack
    X = hstack([X_text, X_cat, X_num])

    return X