import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

S0 = pd.read_csv(r"D:\py\rainfalldata.csv")


#STEP 1 1-DATA CLEANING
print(S0.head(10))
print(S0.info())
print(S0.describe())

#DATA CLEANING
print(S0.isnull().sum())
S0.fillna(S0.mean(numeric_only=True), inplace=True)
print(S0.isnull().sum())
S0.drop_duplicates(inplace=True)
print(S0.info())

#STEP 2- Feature Adding and Engg.
monthly_cols = ['JAN','FEB','MAR','APR','MAY','JUN','JUL','AUG','SEP','OCT','NOV','DEC']
S0['TOTAL_CALCULATED'] = S0[monthly_cols].sum(axis=1)
S0['DIFFERENCE'] = S0['ANNUAL'] - S0['TOTAL_CALCULATED']

print(S0['DIFFERENCE'].describe())
S0['ANNUAL'] = np.where(abs(S0['DIFFERENCE']) > 100,S0['TOTAL_CALCULATED'],S0['ANNUAL'])

# Rainfall intensity category
def rainfall_label(x):
    if x < 500:
        return "Low"
    elif x < 1500:
        return "Medium"
    else:
        return "High"

S0['RAIN_CATEGORY'] = S0['ANNUAL'].apply(rainfall_label)


#STEP 3- VISULIZATION
top7 = S0.groupby('SUBDIVISION')['ANNUAL'].mean().sort_values(ascending=False).head(7)
top7_df = top7.reset_index()
top7_df = top7_df.sort_values(by='ANNUAL')
plt.clf()
plt.figure(figsize=(10,5))
sns.barplot(data=top7_df,x='ANNUAL',y='SUBDIVISION',color='Red')
plt.title("Top 7 High Rainfall States (Flood Risk Zones)")
plt.xlabel("Average Annual Rainfall")
plt.ylabel("State")
plt.show()


yearly_avg = S0.groupby('YEAR')['ANNUAL'].mean()
plt.figure(figsize=(10,5))
sns.lineplot(x=yearly_avg.index, y=yearly_avg.values)
plt.title("Average Annual Rainfall Trend")
plt.show()



plt.figure(figsize=(8,5))
sns.histplot(S0['ANNUAL'], bins=30, kde=True)
plt.title("Distribution of Annual Rainfall")
plt.xlabel("Rainfall")
plt.ylabel("Frequency")
plt.show()

season_cols = ['Jan-Feb','Mar-May','Jun-Sep','Oct-Dec']
season_mean = S0[season_cols].mean()
plt.figure(figsize=(8,4))
sns.barplot(x=season_mean.index,y=season_mean.values,palette='viridis')
plt.title("Average Seasonal Rainfall")
plt.ylabel("Rainfall")
plt.xlabel("Season")
plt.show()



monthly_avg = S0[monthly_cols].mean()
plt.figure(figsize=(8,4))
sns.lineplot(x=monthly_avg.index, y=monthly_avg.values, marker='o')
plt.title("Average Monthly Rainfall Pattern")
plt.xlabel("Month")
plt.ylabel("Rainfall")
plt.show()


sns.countplot(x='RAIN_CATEGORY', data=S0)
plt.title("Rainfall Category Distribution")
plt.show()


matrix = S0[['ANNUAL', 'Jan-Feb', 'Mar-May', 'Jun-Sep', 'Oct-Dec']].corr()
plt.figure(figsize=(6, 4))
sns.heatmap(matrix,annot=True,cmap='coolwarm',linewidths=0.5)
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.title('Heatmap of Annual and Seasonal Rainfall')
plt.show()


# STEP 4 - MACHINE LEARNING
plt.figure(figsize=(6,4))
sns.scatterplot(x=S0['Jun-Sep'], y=S0['ANNUAL'])
plt.title("Jun-Sep vs Annual Rainfall")
plt.xlabel("Monsoon Rainfall")
plt.ylabel("Annual Rainfall")
plt.show()



# STEP 1: Copy cleaned data into S1 (IMPORTANT)
S1 = S0.copy()
S1 = S1.sort_values(['SUBDIVISION','YEAR'])
S1['NEXT_YEAR_RAIN'] = S1.groupby('SUBDIVISION')['ANNUAL'].shift(-1)
S1.dropna(inplace=True)

cols = ['ANNUAL','Jan-Feb','Mar-May','Jun-Sep','Oct-Dec']
X = S1[cols]
y = S1['NEXT_YEAR_RAIN']

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=34)


scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
model = LinearRegression()
model.fit(x_train, y_train)

new_data = pd.DataFrame({
    'ANNUAL':[1500],
    'Jan-Feb':[50],
    'Mar-May':[200],
    'Jun-Sep':[900],
    'Oct-Dec':[300]
})

new_scaled = scaler.transform(new_data)
result = model.predict(new_scaled)
print("Predicted Next Year Rainfall:", result[0])

y_pred = model.predict(x_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error (MSE): {mse:.4f}")


















