I have created a train and test datasets:

```
filtered = data[data["average_bill"].notna() & (data["average_bill"] <= 2500)]

clean_data_train, clean_data_test = train_test_split(
    filtered, stratify=filtered['average_bill'], test_size=0.33, random_state=42)
```

Where `data.head()` command gives next output:

```
	org_id	city	average_bill	rating	rubrics_id	features_id
0	15903868628669802651	msk	1500.0	4.270968	30776 30774	3501685156 3501779478 20422 3502045016 3502045...
1	16076540698036998306	msk	500.0	4.375000	30771	1509 1082283206 273469383 10462 11617 35017794...
2	8129364761615040323	msk	500.0	4.000000	31495	10462 11177 11617 11629 1416 1018 11704 11867 ...
3	15262729117594253452	msk	500.0	4.538813	30776 30770	3501618484 2020795524 11629 11617 1018 11704 2...
4	13418544315327784420	msk	500.0	4.409091	31495	11617 10462 11177 1416 11867 3501744275 20282 ...

```

Now I want to use a CatBoostClassifier on a bloated version of these datasets, named `sparse_data_train` and `sparse_data_test` as follows:

```
clf = CatBoostClassifier()
clf.fit(sparse_data_train, clean_data_train['average_bill'])
test_preds = clf.predict(sparse_data_test)
bal_acc_test = balanced_accuracy_score(sparse_data_test["average_bill"], test_preds)
```

How to get these datasets `sparse_data_train` and `sparse_data_test` with Python code according to the next rule?

```
Вам нужно будет превратить обучающие и тестовые данные в разреженные матрицы sparse_data_train и sparse_data_test соответственно, таким образом, что:

столбец city превратится в столбец из единиц и нулей (например, 1 - msk, 0 - spb);
столбец rating перекочует в разреженные матрицы без изменений;
каждый тип из rubrics_id и каждый тип из features_id превратятся в отдельный 0-1 признак;

В тестовой выборке будут фичи, которых в обучающей выборке не было. Надо создать дополнительную фантомную фичу feature_other, в которой будет то, сколько неизвестных по обучающей выборке фичей есть у данного объекта.
```
