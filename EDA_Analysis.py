import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelBinarizer
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV


from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Read the CSV file into a pandas dataframe
df = pd.read_csv('/home/jeeva/Projects/python/Copper_Model/data/Copper_Set_Result.csv')
print(df.head(2))
print(len(df['customer'].unique())) 
#print(df.shape)
#print(df.info())

# missing values
print("\nBefore  : Missing_values_count")
missing_values_count = df.isnull().sum()
#print(missing_values_count)


# dealing with data in wrong format
df['item_date'] = pd.to_datetime(df['item_date'], format='%Y%m%d', errors='coerce').dt.date
df['quantity tons'] = pd.to_numeric(df['quantity tons'], errors='coerce')
df['customer'] = pd.to_numeric(df['customer'], errors='coerce')
df['country'] = pd.to_numeric(df['country'], errors='coerce')
df['application'] = pd.to_numeric(df['application'], errors='coerce')
df['thickness'] = pd.to_numeric(df['thickness'], errors='coerce')
df['width'] = pd.to_numeric(df['width'], errors='coerce')
df['material_ref'] = df['material_ref'].str.lstrip('0')
df['product_ref'] = pd.to_numeric(df['product_ref'], errors='coerce')
df['delivery date'] = pd.to_datetime(df['delivery date'], format='%Y%m%d', errors='coerce').dt.date
df['selling_price'] = pd.to_numeric(df['selling_price'], errors='coerce')


print("\n\nAfter  : Missing_values_count")
missing_values_count = df.isnull().sum()
#print(missing_values_count)
#print(df.shape)
#print(df.info())

# material_ref has large set of null values, so replacing those with unknown 
df['material_ref'].fillna('unknown', inplace=True)
# deleting the remaining null values as they are less than 1% of data which can be neglected
df = df.dropna()

missing_values_count = df.isnull().sum()
#print(missing_values_count)
#print(df.shape)


df_p=df.copy()


mask1 = df_p['selling_price'] <= 0
print(mask1.sum())
df_p.loc[mask1, 'selling_price'] = np.nan

mask1 = df_p['quantity tons'] <= 0
print(mask1.sum())
df_p.loc[mask1, 'quantity tons'] = np.nan

mask1 = df_p['thickness'] <= 0
print(mask1.sum())

print(df_p.isnull().sum())

df_p.dropna(inplace=True)
print(df_p.shape)

#Checking Skewness
#sns.distplot(df_p['quantity tons']) # Skewed, ie :Data is not distributed properly, so need to fix the data
#plt.show()

#sns.distplot(df_p['country']) # OK
#plt.show()

#sns.distplot(df_p['application']) # OK
#plt.show()

#sns.distplot(df_p['thickness']) # Skewed,
#plt.show()

#sns.distplot(df_p['width']) # OK
#plt.show()

#sns.distplot(df_p['selling_price']) # Skewed,
#plt.show()


# Change the data of quantity tons, thickness and selling_price
df_p['quantity tons_log'] = np.log(df_p['quantity tons'])
sns.distplot(df_p['quantity tons_log'])
#plt.show()

df_p['thickness_log'] = np.log(df_p['thickness'])
sns.distplot(df_p['thickness_log'])
#plt.show()

df_p['selling_price_log'] = np.log(df_p['selling_price'])
sns.distplot(df_p['selling_price_log'])
#plt.show()


X=df_p[['quantity tons_log','status','item type','application','thickness_log','width','country','customer','product_ref']]
y=df_p['selling_price_log']


# encoding categorical variables
ohe_type = OneHotEncoder(handle_unknown='ignore')
ohe_type.fit(X[['item type']])
X_ohe_item_type = ohe_type.fit_transform(X[['item type']]).toarray()
ohe_status = OneHotEncoder(handle_unknown='ignore')
ohe_status.fit(X[['status']])
X_ohe_status = ohe_status.fit_transform(X[['status']]).toarray()


# independent features after encoding
X = np.concatenate((X[['quantity tons_log', 'application', 'thickness_log', 'width','country','customer','product_ref']].values, X_ohe_item_type, X_ohe_status), axis=1)
scaler = StandardScaler()
X = scaler.fit_transform(X)


# test and train split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)


# Decision tree Regression
dtr = DecisionTreeRegressor()
# hyperparameters
param_grid = {'max_depth': [2, 5, 10, 20],
              'min_samples_split': [2, 5, 10],
              'min_samples_leaf': [1, 2, 4],
              'max_features': ['auto', 'sqrt', 'log2']}

# gridsearchcv
grid_search = GridSearchCV(estimator=dtr, param_grid=param_grid, cv=5)
grid_search.fit(X_train, y_train)


best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
print(y_pred)

# evalution metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print('Mean squared error:', mse)
print('R-squared:', r2)


# Saving the model
import pickle
with open('/home/jeeva/Projects/python/Copper_Model/model.pkl', 'wb') as file:
    pickle.dump(best_model, file)
with open('/home/jeeva/Projects/python/Copper_Model/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
with open('/home/jeeva/Projects/python/Copper_Model/ohe_type.pkl', 'wb') as f:
    pickle.dump(ohe_type, f)
with open('/home/jeeva/Projects/python/Copper_Model/ohe_status.pkl', 'wb') as f:
    pickle.dump(ohe_status, f)
    
    
    
#Classification
df_c = df_p[df_p['status'].isin(['Won', 'Lost'])]
print(len(df_c))

Y = df_c['status']
X= df_c[['quantity tons_log','selling_price_log','item type','application','thickness_log','width','country','customer','product_ref']]

# encoding categorical variables
ohe = OneHotEncoder(handle_unknown='ignore')
ohe.fit(X[['item type']])
X_ohe = ohe.fit_transform(X[['item type']]).toarray()

be = LabelBinarizer()
be.fit(Y) 
y = be.fit_transform(Y)

# independent features after encoding
X = np.concatenate((X[['quantity tons_log', 'selling_price_log','application', 'thickness_log', 'width','country','customer','product_ref']].values, X_ohe), axis=1)
scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# decision tree classifier
dtc = DecisionTreeClassifier()
dtc.fit(X_train, y_train)
y_pred = dtc.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")
cm = confusion_matrix(y_test, y_pred)
print(f"Confusion Matrix:\n{cm}")


# Saving the model
import pickle
with open('/home/jeeva/Projects/python/Copper_Model/cmodel.pkl', 'wb') as file:
    pickle.dump(dtc, file)
with open('/home/jeeva/Projects/python/Copper_Model/cscaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
with open('/home/jeeva/Projects/python/Copper_Model/c_ohe_type.pkl', 'wb') as f:
    pickle.dump(ohe, f)