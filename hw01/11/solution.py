import pandas as pd
from catboost import CatBoostClassifier
from scipy.sparse import csr_matrix, hstack

def build_sparse(train_df, test_df):
    # --- 1. City encoding (binary msk=1, spb=0)
    city_train = (train_df['city'] == 'msk').astype(int).values.reshape(-1, 1)
    city_test = (test_df['city'] == 'msk').astype(int).values.reshape(-1, 1)

    # --- 2. Rating (keep numeric)
    rating_train = train_df[['rating']].values
    rating_test = test_df[['rating']].values

    # --- 3. Expand rubrics_id and features_id into sets
    train_rubrics = train_df['rubrics_id'].str.split()
    train_features = train_df['features_id'].str.split()
    test_rubrics = test_df['rubrics_id'].str.split()
    test_features = test_df['features_id'].str.split()

    # Collect unique IDs from training
    rubrics_vocab = sorted(set().union(*train_rubrics))
    features_vocab = sorted(set().union(*train_features))

    # Index maps
    rubric_to_idx = {r: i for i, r in enumerate(rubrics_vocab)}
    feature_to_idx = {f: i for i, f in enumerate(features_vocab)}

    # --- 4. Encode train rubrics/features
    rows, cols = [], []
    for i, rubs in enumerate(train_rubrics):
        for r in rubs:
            if r in rubric_to_idx:
                rows.append(i)
                cols.append(rubric_to_idx[r])
    rubrics_train = csr_matrix(( [1]*len(rows), (rows, cols) ), shape=(len(train_df), len(rubrics_vocab)))

    rows, cols = [], []
    for i, feats in enumerate(train_features):
        for f in feats:
            if f in feature_to_idx:
                rows.append(i)
                cols.append(feature_to_idx[f])
    features_train = csr_matrix(( [1]*len(rows), (rows, cols) ), shape=(len(train_df), len(features_vocab)))

    # --- 5. Encode test rubrics/features
    rows, cols = [], []
    for i, rubs in enumerate(test_rubrics):
        for r in rubs:
            if r in rubric_to_idx:  # known rubric
                rows.append(i)
                cols.append(rubric_to_idx[r])
    rubrics_test = csr_matrix(( [1]*len(rows), (rows, cols) ), shape=(len(test_df), len(rubrics_vocab)))

    rows, cols = [], []
    feature_other = [0]*len(test_df)
    for i, feats in enumerate(test_features):
        for f in feats:
            if f in feature_to_idx:  # known feature
                rows.append(i)
                cols.append(feature_to_idx[f])
            else:  # unseen feature → count in feature_other
                feature_other[i] += 1
    features_test = csr_matrix(( [1]*len(rows), (rows, cols) ), shape=(len(test_df), len(features_vocab)))
    feature_other = csr_matrix(pd.DataFrame(feature_other))

    # --- 6. Stack everything together
    sparse_data_train = hstack([
        csr_matrix(city_train),
        csr_matrix(rating_train),
        rubrics_train,
        features_train
    ])

    sparse_data_test = hstack([
        csr_matrix(city_test),
        csr_matrix(rating_test),
        rubrics_test,
        features_test,
        feature_other  # phantom column
    ])

    return sparse_data_train, sparse_data_test

sparse_data_train, sparse_data_test = build_sparse(clean_data_train, clean_data_test)

clf = CatBoostClassifier()
clf.fit(sparse_data_train, clean_data_train['average_bill'])

test_preds = clf.predict(sparse_data_test)
catb_acc_test = balanced_accuracy_score(clean_data_test["average_bill"], test_preds)

print("Test Balanced Accuracy:", catb_acc_test)