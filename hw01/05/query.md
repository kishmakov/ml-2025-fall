
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

And I want to get a simple model predicting mean value of `average_bill` column
motivated by sklearn.baseRegressorMixin.

Can you implement it?

