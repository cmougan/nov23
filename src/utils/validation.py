def train_test_split_temporal(
    X, y, date_col: str = "date", date_split: str = "2019-01-01"
):
    X_tr = X[X[date_col] < date_split]
    X_te = X[X[date_col] >= date_split]
    y_tr = y[X[date_col] < date_split]
    y_te = y[X[date_col] >= date_split]
    return X_tr, X_te, y_tr, y_te
