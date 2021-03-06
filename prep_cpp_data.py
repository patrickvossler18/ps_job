import pandas as pd
from itertools import chain

def clean_cpp_data(X):
    factor_list = pd.read_csv("~/ps_job/factor_list.csv").values.tolist()

    factor_list = list(chain(*factor_list))
    # Drop Y and W
    X_new = X
    # Convert factors to dummies
    chunk_list = []
    for factor in factor_list:
        # expand the variable
        expanded = pd.get_dummies(data=X[factor])
        # count how many columns
        chunks = expanded.shape[1]
        chunk_list.append(chunks)
        X_new = pd.concat([X_new, expanded], axis=1)

    X_new = X_new.drop(columns = factor_list)
    return([X_new,chunk_list])
