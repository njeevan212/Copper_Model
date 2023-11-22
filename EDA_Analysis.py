import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

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
plt.show()

df_p['thickness_log'] = np.log(df_p['thickness'])
sns.distplot(df_p['thickness_log'])
plt.show()

df_p['selling_price_log'] = np.log(df_p['selling_price'])
sns.distplot(df_p['selling_price_log'])
plt.show()
