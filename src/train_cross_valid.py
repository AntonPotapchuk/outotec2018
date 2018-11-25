import lightgbm as lgb
import pandas as pd
import pickle
import sys
from sklearn.model_selection import KFold, train_test_split, ParameterGrid
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

RANDOM_STATE = 1234


def train_cross_valid(X, y):
    i = 0
    log = []

    gridParams = {
        'n_estimators': [50, 150],
        'num_leaves': [6, 8, 16],
        'boosting_type': ['gbdt'],
        'objective': ['binary'],
        'random_state': [RANDOM_STATE],
        'reg_alpha': [0, 1],
        'reg_lambda': [0, 1],
        'class_weight': ['balanced'],
        'learning_rate': [0.005, 0.001, 0.01]
    }

    kf = KFold(n_splits=5, random_state=RANDOM_STATE, shuffle=False)
    for ind_train, ind_test in kf.split(X):
        i += 1
        X_train, X_val, y_train, y_val = train_test_split(X.iloc[ind_train], y.iloc[ind_train], test_size=0.25,
                                                          shuffle=False)
        X_test = X.iloc[ind_test]
        y_test = y.iloc[ind_test]
        print("Fold %d" % i)

        best_params = None
        best_train_acc = 0
        best_val_acc = 0

        for params in ParameterGrid(gridParams):
            model = lgb.LGBMClassifier(**params)
            model.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=30, verbose=False)
            train_acc = accuracy_score(y_train, model.predict(X_train))
            val_acc = accuracy_score(y_val, model.predict(X_val))
            test_acc = accuracy_score(y_test, model.predict(X_test))
            if val_acc > best_val_acc:
                best_params = model.get_params()
                best_train_acc = train_acc
                best_val_acc = val_acc
        X_train = X.iloc[ind_train]
        y_train = y.iloc[ind_train]
        model = lgb.LGBMClassifier(**best_params)
        model.fit(X_train, y_train, eval_set=(X_train, y_train), early_stopping_rounds=30, verbose=False)
        test_acc = accuracy_score(y_test, model.predict(X_test))
        log.append({'train_acc': best_train_acc,
                    'val_acc': best_val_acc,
                    'test_acc': test_acc,
                    'params': best_params
                    })
        print(log[-1])
    return log


if __name__ == "__main__":
    # check the number of arguments
    if len(sys.argv) != 3:
        print("Usage: generate_dataset <input_dir> <output_dir>", file=sys.stderr)
        exit(-1)

    input_folder = sys.argv[1]
    output_folder = sys.argv[2]

    full_df = pd.read_csv('full_df_15s.csv', parse_dates=[0])
    encoder = LabelEncoder()
    full_df['status'] = encoder.fit_transform(full_df['status'])
    log = train_cross_valid(full_df.iloc[:, 2:], full_df['status'])
    pickle.dump(log, open(output_folder, 'wb'))
