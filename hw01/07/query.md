I am working with `data` created from pandas table as follows:

```
data = pd.read_csv(base + 'organisations.csv')
```

I cleaned data as follows:

```
filtered = data[data["average_bill"].notna() & (data["average_bill"] <= 2500)]
```

Now I have split this data into train and test sets:

```
clean_data_train, clean_data_test = train_test_split(
    filtered, stratify=filtered['average_bill'], test_size=0.33, random_state=42)
```

`MeanRegressor` is a regressor (motivated by `sklearn.base.ClassifierMixin`), which predicts avalue of `average_bill` column, showing the most common value for the column.

For `CityMeanRegressor` I want to use `city` feature, which can be either `msk` or `spb`, to predict value more accurately.

Can you implement it?
