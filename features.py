import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the uploaded CSV file
file_path = r"C:\Users\hirun\Downloads\social_media_preprossed (1).csv"
df = pd.read_csv(file_path)

# -----------------------------
# 1) Category with Tags
# -----------------------------
if df['tags'].dtype == 'object':
    df['tags_split'] = df['tags'].fillna("").astype(str).str.split(",")
else:
    df['tags_split'] = df['tags']

df_exploded = df.explode('tags_split')
df_exploded['tags_split'] = df_exploded['tags_split'].str.strip()

category_tags = df_exploded.groupby(['category', 'tags_split']).size().reset_index(name='count')

# -----------------------------
# 2) Category with Region
# -----------------------------
category_region = df.groupby(['category', 'region']).size().reset_index(name='count')

# -----------------------------
# 3) Post Day with Region
# -----------------------------
postday_region = df.groupby(['post_day', 'region']).size().reset_index(name='count')

# -----------------------------
# Visualization Section
# -----------------------------

sns.set(style="whitegrid")

# 1) Top 15 Tags per Category (heatmap style)
pivot_tags = category_tags.pivot_table(index="category", columns="tags_split", values="count", fill_value=0)
plt.figure(figsize=(14, 6))
sns.heatmap(pivot_tags.iloc[:, :15], cmap="YlGnBu", annot=False, cbar=True)  # Only first 15 tags for readability
plt.title("Top Tags Distribution across Categories", fontsize=16)
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()

# 2) Category vs Region (stacked bar chart)
category_region_pivot = category_region.pivot(index="category", columns="region", values="count").fillna(0)
category_region_pivot.plot(kind="bar", stacked=True, figsize=(12, 6), cmap="tab20")
plt.title("Posts by Category and Region", fontsize=16)
plt.ylabel("Count of Posts")
plt.xlabel("Category")
plt.legend(title="Region", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()


# 3) Post Day vs Region (grouped bar chart)
plt.figure(figsize=(12, 6))
sns.barplot(data=postday_region, x="post_day", y="count", hue="region", palette="Set2")
plt.title("Posts by Day of the Week and Region", fontsize=16)
plt.ylabel("Count of Posts")
plt.xlabel("Day of Week")
plt.legend(title="Region", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()
