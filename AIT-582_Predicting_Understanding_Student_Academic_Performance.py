#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd  # Importing pandas 


df = pd.read_csv('StudentPerformance.csv')
 # Display the first few rows of the DataFrame
df.head()  


# In[3]:


# Check for duplicates
duplicates = df.duplicated()
print("Number of duplicate rows: ",duplicates.sum())
if duplicates.sum() >0:
  print("\nDuplicate rows: ")
  print(df[duplicates])


# In[3]:


# Check for missing values
df.isnull().sum()


# In[4]:


# Check for empty strings or whitespace-only values
empty_strings = (df == '') | (df.applymap(lambda x: isinstance(x, str) and x.isspace()))
empty_strings_count = empty_strings.sum()
print("\nNumber of empty strings per column values per column: ")
print(empty_strings_count[empty_strings_count>0])


# In[5]:


# Check for 'NaN' or 'null' strings
nan_strings = (df == 'NaN') | (df == 'null')
nan_strings_count = nan_strings.sum()
print("\nNumber of 'NaN' or 'null' strings per column: ")
print(nan_strings_count[nan_strings_count>0])


# In[7]:


# Check for columns with mixed data types
mixed_dtypes = df.applymap(type).nunique()
print("\nColumns with mixed data types: ")
print(mixed_dtypes[mixed_dtypes>1])


# In[ ]:


# Handle missing values - Example: Fill missing values with a placeholder or statistical measure
df['Teacher_Quality'] = df['Teacher_Quality'].fillna('Unknown')  # Filling with 'Unknown' for categorical data
df['Parental_Education_Level'] = df['Parental_Education_Level'].fillna('Unknown')
df['Distance_from_Home'] = df['Distance_from_Home'].fillna(df['Distance_from_Home'].mode()[0])  # Filling with mode


# In[2]:


df.info()  # Displays column names, data types, and counts of non-null values



# In[11]:


# Print summary of changes
print("\nCleaned dataset: ")
print(df.head())


# In[ ]:


# List of columns to convert to categorical data types
categorical_cols = [
    'Parental_Involvement', 'Access_to_Resources', 'Extracurricular_Activities',
    'Motivation_Level', 'Internet_Access', 'Family_Income', 'Teacher_Quality',
    'School_Type', 'Peer_Influence', 'Learning_Disabilities',
    'Parental_Education_Level', 'Distance_from_Home', 'Gender'


# In[4]:


# Display unique value counts for 'Teacher_Quality'
print(df['Teacher_Quality'].value_counts())

# Display unique value counts for 'Parental_Education_Level'
print(df['Parental_Education_Level'].value_counts())

# Display unique value counts for 'Distance_from_Home'
print(df['Distance_from_Home'].value_counts())


# In[5]:


# Fill missing values
df['Teacher_Quality'].fillna('Medium', inplace=True)
df['Parental_Education_Level'].fillna('High School', inplace=True)
df['Distance_from_Home'].fillna('Near', inplace=True)


# In[8]:


# Convert specific columns to categorical data types
categorical_cols = [
    'Parental_Involvement', 'Access_to_Resources', 'Extracurricular_Activities',
    'Motivation_Level', 'Internet_Access', 'Family_Income', 'Teacher_Quality',
    'School_Type', 'Peer_Influence', 'Learning_Disabilities',
    'Parental_Education_Level', 'Distance_from_Home', 'Gender'
]

df[categorical_cols] = df[categorical_cols].astype('category')


# In[9]:


# Display unique values in each categorical column
for col in categorical_cols:
    print(f"Unique values in {col}:\n{df[col].unique()}\n")


# In[6]:


#Does Gender or School_Type affect Exam_Score?
# Mean Exam_Score by Gender
print(df.groupby('Gender')['Exam_Score'].mean())

# Mean Exam_Score by School_Type
print(df.groupby('School_Type')['Exam_Score'].mean())


# In[10]:


#What is the distribution of exam scores? Are there any noticeable peaks or outliers in the data?

import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10, 6))
sns.histplot(df['Exam_Score'], bins=20, kde=True)
plt.title('Distribution of Exam Scores')
plt.xlabel('Exam Score')
plt.ylabel('Frequency')
plt.show()


# In[11]:


#What are the key factors that influence students' exam performance, and how do variables such as attendance, hours studied, 
#and prior academic performance contribute to their success?
# Correlation matrix
print("Correlation matrix:")
print(df.corr())

# Heatmap visualization of correlation
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()


# In[7]:


# What is the relationship between extracurrricular activities and sleep hours?
plt.figure(figsize=(10, 6))
sns.histplot(data=df, x='Sleep_Hours', hue='Extracurricular_Activities', kde=True, bins=10, palette='Set2')
plt.title('Distribution of Sleep Hours by Extracurricular Activities')
plt.xlabel('Sleep Hours')
plt.ylabel('Count')
plt.legend(title='Extracurricular Activities', labels=df['Extracurricular_Activities'].unique())
plt.show()


# In[14]:


#How do different numerical features correlate with each other, and
#what patterns or clusters can be observed across the various features 
#in relation to Exam Score?

# Create a pairplot to visualize relationships between numerical features
sns.pairplot(df, vars=['Hours_Studied', 'Attendance', 'Previous_Scores', 'Tutoring_Sessions', 'Physical_Activity', 'Sleep_Hours', 'Exam_Score'], hue='Exam_Score', palette='coolwarm', markers='o', diag_kind='kde')
plt.suptitle('Pairplot of Numerical Features', y=1.02)  # Add a title with adjustment
plt.show()


# In[6]:


#How does peer influence interact with physical activity levels to impact exam scores?

# Group data by Peer Influence and Physical Activity levels, then calculate mean Exam Scores
grouped = df.groupby(['Peer_Influence', 'Physical_Activity'])['Exam_Score'].mean().unstack()

# Create a bar plot
grouped.plot(kind='bar', figsize=(12, 6), colormap='plasma')

# Customize the plot
plt.title('Impact of Peer Influence and Physical Activity on Exam Scores', fontsize=14)
plt.ylabel('Average Exam Score', fontsize=12)
plt.xlabel('Peer Influence Type', fontsize=12)
plt.xticks(rotation=0)
plt.legend(title='Physical Activity Level', fontsize=10)
plt.tight_layout()

# Display the plot
plt.show()


# In[5]:


# What are the ingluence of internet access and motivation level on student performane?

# Grouped bar chart: Exam_Score by Internet_Access and Motivation_Level
plt.figure(figsize=(10, 6))
sns.barplot(data=df, x="Motivation_Level", y="Exam_Score", hue="Internet_Access", ci=None, palette="coolwarm")
plt.title("Impact of Internet Access and Motivation on Exam Scores")
plt.xlabel("Motivation Level")
plt.ylabel("Average Exam Score")
plt.legend(title="Internet Access")
plt.show()

# Heatmap to check interaction between Internet_Access and Motivation_Level
pivot_table = df.pivot_table(values="Exam_Score", index="Motivation_Level", columns="Internet_Access", aggfunc="mean")
sns.heatmap(pivot_table, annot=True, cmap="YlGnBu")
plt.title("Heatmap: Internet Access vs Motivation Level and Exam Scores")
plt.xlabel("Internet Access")
plt.ylabel("Motivation Level")
plt.show()



# In[3]:


# What Impact can peer influence and school type have on exam scores.
import seaborn as sns
import matplotlib.pyplot as plt

# Boxplot to compare Exam_Score across Peer_Influence and School_Type
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x="Peer_Influence", y="Exam_Score", hue="School_Type")
plt.title("Impact of Peer Influence and School Type on Exam Scores")
plt.xlabel("Peer Influence")
plt.ylabel("Exam Score")
plt.legend(title="School Type")
plt.show()


# In[29]:


# Gradient Boosting Regressor with hyperparameter tuning.
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Define the parameter grid
param_grid = {
    'n_estimators': [50, 100, 150],
    'learning_rate': [0.05, 0.1, 0.2],
    'max_depth': [3, 4, 5]
}

# Initialize the Gradient Boosting Regressor
gb_model = GradientBoostingRegressor(random_state=42)

# Set up GridSearchCV
grid_search = GridSearchCV(estimator=gb_model, param_grid=param_grid, cv=5, scoring='r2', n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

# Best parameters from grid search
best_params = grid_search.best_params_
print(f"Best Parameters: {best_params}")

# Train the Gradient Boosting model with the best parameters
best_gb_model = grid_search.best_estimator_

# Make predictions on the test set
y_pred = best_gb_model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error (MSE): {mse}")
print(f"R-squared (R2): {r2}")

# Plotting the predicted vs. actual exam scores for the test set
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.6)
plt.plot([y.min(), y.max()], [y.min(), y.max()], color='red', linestyle='--', linewidth=2)
plt.xlabel("Actual Exam Scores")
plt.ylabel("Predicted Exam Scores")
plt.title("Tuned Gradient Boosting: Actual vs Predicted Exam Scores")
plt.show()


# In[88]:


#What effects does parental participation have on students academic performance?

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

# Prepare data
X = pd.get_dummies(df[['Parental_Involvement']], drop_first=True)
y = df['Exam_Score']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

# Calculate metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Plot: Actual vs. Predicted Exam Scores
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, color='purple', alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--', linewidth=2)
plt.title("Actual vs. Predicted Exam Scores (Random Forest Regressor)")
plt.xlabel("Actual Exam Scores")
plt.ylabel("Predicted Exam Scores")
plt.text(60, 90, f'MSE: {mse:.2f}\nR2: {r2:.2f}', fontsize=12, color='black', bbox=dict(facecolor='white', alpha=0.5))
plt.grid(True)
plt.show()


# In[7]:


#What part do extracurricular activities and resource availability have in a students academic success?

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
import matplotlib.pyplot as plt
import seaborn as sns

# Prepare data
X = pd.get_dummies(df[['Extracurricular_Activities', 'Access_to_Resources']], drop_first=True)
y = df['Exam_Score']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model
svr = SVR(kernel='linear')
svr.fit(X_train, y_train)
y_pred = svr.predict(X_test)

# Add predicted scores and corresponding actual scores to a DataFrame for plotting
plot_data = pd.DataFrame({'Actual Exam Score': y_test, 'Predicted Exam Score': y_pred, 'Access_to_Resources': df['Access_to_Resources'].iloc[y_test.index]})


# Linear Fit Plot with Seaborn
plt.figure(figsize=(8, 6))
sns.regplot(x='Actual Exam Score', y='Predicted Exam Score', data=plot_data, scatter_kws={'alpha':0.6, 'color':'orange'}, line_kws={'color':'blue'})
plt.title("Linear Fit Plot: Actual vs Predicted Exam Scores (SVR)")
plt.xlabel("Actual Exam Scores")
plt.ylabel("Predicted Exam Scores")
plt.grid(True)
plt.show()





# In[ ]:




