# ML LAB1: Data Exploration and Preprocessing

Work pair : SACI Zhour & ABDELKRIM Jazia



## Goal
Learn to load, explore, and clean data before using it in a machine learning model.
 
## Libraries
`pandas`, `scikit-learn`, `seaborn`, `matplotlib`



### Exercise 1: Iris dataset (tables and splits)
- Load the CSV into a DataFrame `D`.
- Look at it with `info()`, `describe()`, `shape`, `columns`, `head()`.
- Pick two features (`sepal_length`, `petal_width`) and show the first 20 rows.
- Make a copy of the data (so the original stays safe).
- Split into 80% train / 20% test (`random_state=42`, `stratify=y`).
- Check the sizes of both sets.




### Exercise 2: Text data
- Read `txt_file.txt` (UTF-8) line by line.
- Load the 20 Newsgroups dataset.
- Show the number of documents, labels, the first document, and class names.





### Exercise 3: Missing values
- Load `employees.csv` and count missing values (`isna().sum()`).
- Plot them with a seaborn heatmap.
- Add a fake row with `'-'` and `'na'`, then convert these to `np.nan`.
- Fill gaps with `SimpleImputer` (median for numbers, most frequent for text).
- Compare the counts before and after.



### Exercise 4: Scaling and encoding
- **Scaling:** try `StandardScaler`, `MinMaxScaler`, `MaxAbsScaler`, `RobustScaler`, `normalize` on a small matrix, then binarize it.
- **Encoding:** use `OrdinalEncoder` on the features, `LabelEncoder` on the target, and `OneHotEncoder` on the target.