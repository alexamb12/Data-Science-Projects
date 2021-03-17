import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('insurance.csv')

# print(df.head())

# Distribution of insurance costs amongst smokers/non-smokers using BMI
sns.set_palette('pastel')

f, ax = plt.subplots(figsize=(8, 6))
ax = sns.swarmplot(x=df['smoker'], y=df['charges'], size = 3)
ax.set_xlabel("Smoker")
ax.set_ylabel("Insurane Charges")
ax.set_title("Average Cost of Insurance Charges in Smokers/Non-Smokers Using BMI")

sns.despine()
# plt.show()
plt.close()

# insurance cost between women & men with no children

df_children = df[df['children'] == 0]
# print(df_children.head())

df1 = df_children.groupby(['sex', 'smoker'])['charges'].mean().reset_index()
# print(df1)

df1_pivot = df1.pivot(
    columns = 'smoker', 
    index= 'sex', 
    values = 'charges'
)
# print(df1_pivot)

plt.figure(1)
f3, ax3 = plt.subplots(figsize=(10,8))
ax3 = sns.barplot(x='sex', y= 'charges', hue= 'smoker', data = df1)
ax3.set_title("Average Insurance Costs Between Men & Women with No Children (Smokers vs. Non-Smokers")
# plt.show()

# plt.close()

plt.figure(2)
f4,ax4 = plt.subplots(figsize=(10,8))
ax4 = sns.swarmplot(x='sex', y='charges', hue='smoker', data=df_children, size=3.5)
ax4.set_xlabel("Sex")
ax4.set_ylabel("Insurance Charges (USD $)")
ax4.set_title("Distribution of Insurance Charges Among Men & Women with No Children Using BMI (Smokers/Non-Smokers)")
plt.show()
