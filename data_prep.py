import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the large dataset
df = pd.read_csv('data/student_performance_dataset_bigger.csv')


# 1. Plotting the Score Distribution
plt.figure(figsize=(10, 6))
sns.histplot(df['exam_score'], bins=50, kde=True, color='royalblue')
plt.axvline(55, color='red', linestyle='--', linewidth=2, label='Target Goal (55%)')
plt.title('Current Student Performance Distribution (N=1,000,000)')
plt.xlabel('Exam Score (%)')
plt.ylabel('Number of Students')
plt.legend()
plt.savefig('performance_distribution.jpg')

# 2. Plotting the "Drivers of Success"
plt.figure(figsize=(10, 6))
numeric_df = df.select_dtypes(include=['float64', 'int64']).drop(columns=['student_id'])
corrs = numeric_df.corr()['exam_score'].sort_values().drop('exam_score')
corrs.plot(kind='barh', color='skyblue')
plt.title('Correlation: What Factors Predict Exam Success?')
plt.xlabel('Correlation Coefficient')
plt.tight_layout()
plt.savefig('correlation_drivers.jpg')

print("Report generated: Check performance_distribution.jpg and correlation_drivers.jpg")