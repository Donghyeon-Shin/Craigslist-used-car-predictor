import pandas as pd

# Read the CSV file
df = pd.read_csv('./Data/vehicles.csv')

# Display the first few rows of the dataframe
df.head()

# Display the information(non null count, Dtype) about the dataframe
df.info()

# Check for missing values
df.isnull().sum()

# Display the summary statistics of the dataframe
df.describe()

import matplotlib.pyplot as plt
import seaborn as sns

# Plot the distribution of the price ($500 ~ $100,000)
filtered_df = df[(df['price'] >= 500) & (df['price'] <= 100000)]

plt.figure(figsize=(10, 6))
sns.histplot(filtered_df['price'], bins=50, edgecolor='black')

plt.title('Distribution of Used Car Prices ($500 - $100,000)')
plt.xlabel('Price (USD)')
plt.ylabel('Frequency')

# Plot the distribution of the year (boxplot)
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='year', color='coral')
plt.title('Distribution of Used Car Prices by Year')
plt.xlabel('Year')
plt.show()

# 300,000 miles is the realistic upper limit for odometer
filtered_df = df[df['odometer'] <= 300000]

# display the boxplot of the odometer
plt.figure(figsize=(10, 3))
sns.boxplot(data=filtered_df, x='odometer', color='skyblue')

plt.title('Boxplot of Odometer (0 - 300,000 miles)', fontsize=15, fontweight='bold')
plt.xlabel('Odometer (Miles)', fontsize=12)

plt.show()

def show_countplot(df, column):
    # Fill the missing values with 'N/A' to view the distribution of the column
    df_clean = df.copy()
    df_clean[column] = df_clean[column].fillna('N/A')

    # Plot the distribution of the column (countplot)
    plt.figure(figsize=(10, 10))
    # palette='viridis' is a color palette for the plot
    sns.countplot(y=column, data=df_clean, hue=column, order=df_clean[column].value_counts().index, palette='viridis')
    plt.title('Distribution of Used Car ' + column)
    plt.xlabel(column)
    plt.show()

# show the distribution of the manufacturer
show_countplot(df, 'manufacturer')

# show the distribution of the cylinders
show_countplot(df, 'cylinders')

# show the distribution of the fuel
show_countplot(df, 'fuel')

# show the distribution of the transmission
show_countplot(df, 'transmission')

# show the distribution of the Paint Color
show_countplot(df, 'paint_color')

# show the distribution of the condition
show_countplot(df, 'condition')

# show the distribution of the drive 
show_countplot(df, 'drive')
