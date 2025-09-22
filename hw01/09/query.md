What would be a code of such classifier for such a task? Use some classes from `sklearn.base` as a parent classes.


```
Теперь обучите классификатор, который для заведения предсказывает медиану среднего чека по всем объектам тестовой выборки с таким же, как у него, значением modified_features, а если такого в обучающей выборке нет, то глобальную медиану среднего чека по всей обучающей выборке.

```

================================================

I have created a train and test datasets:

```
filtered = data[data["average_bill"].notna() & (data["average_bill"] <= 2500)]

clean_data_train, clean_data_test = train_test_split(
    filtered, stratify=filtered['average_bill'], test_size=0.33, random_state=42)

clean_data_train["modified_features"] = (
    clean_data_train["rubrics_id"].astype(str)
    + " q " +
    clean_data_train["features_id"].astype(str)
)

# множество допустимых комбинаций
train_combos = set(clean_data_train["modified_features"].unique())

# test: склеиваем
clean_data_test["modified_features"] = (
    clean_data_test["rubrics_id"].astype(str)
    + " q " +
    clean_data_test["features_id"].astype(str)
)

# заменяем всё, чего нет в train
clean_data_test.loc[
    ~clean_data_test["modified_features"].isin(train_combos),
    "modified_features"
] = "other"
```

Now I want to run classifier on train, set apply to test set like following:

```
clf = ModifiedFeaturesMedianClassifier()
clf.fit(clean_data_train, clean_data_train["average_bill"])

# 2. Predict on test
preds = clf.predict(clean_data_test)
```

Now I want to generate a simple *.csv file: first column is the index of test item in initial data object (it can be uniqly identified by `org_id` column), second is the value of preds.

How to do it?