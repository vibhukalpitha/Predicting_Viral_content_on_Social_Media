# features.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

def plot_top_tags(df, category=None, region=None, top_n=15):
    df_filtered = df.copy()
    if category:
        df_filtered = df_filtered[df_filtered['category'] == category]
    if region:
        df_filtered = df_filtered[df_filtered['region'] == region]

    if df_filtered['tags'].dtype == 'object':
        df_filtered['tags_split'] = df_filtered['tags'].fillna("").astype(str).str.split(",")
    else:
        df_filtered['tags_split'] = df_filtered['tags']

    df_exploded = df_filtered.explode('tags_split')
    df_exploded['tags_split'] = df_exploded['tags_split'].str.strip()

    category_tags = df_exploded.groupby(['category', 'tags_split']).size().reset_index(name='count')

    pivot_tags = category_tags.pivot_table(index="category", columns="tags_split", values="count", fill_value=0)

    plt.figure(figsize=(14, 6))
    sns.heatmap(pivot_tags.iloc[:, :top_n], cmap="YlGnBu", annot=False, cbar=True)
    plt.title(f"Top Tags Distribution for Category: {category}, Region: {region}", fontsize=16)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    st.pyplot(plt)
    plt.clf()

def plot_category_region(df, category=None):
    df_filtered = df.copy()
    if category:
        df_filtered = df_filtered[df_filtered['category'] == category]

    category_region = df_filtered.groupby(['category', 'region']).size().reset_index(name='count')
    if category_region.empty:
        st.info("No data available for this selection.")
        return

    category_region_pivot = category_region.pivot(index="category", columns="region", values="count").fillna(0)
    ax = category_region_pivot.plot(kind="bar", stacked=True, figsize=(12, 6), cmap="tab20")
    plt.title(f"Posts by Category and Region for Category: {category}", fontsize=16)
    plt.ylabel("Count of Posts")
    plt.xlabel("Category")
    plt.legend(title="Region", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    st.pyplot(plt)
    plt.clf()

def plot_postday_region(df, category=None, region=None):
    df_filtered = df.copy()
    if category:
        df_filtered = df_filtered[df_filtered['category'] == category]
    if region:
        df_filtered = df_filtered[df_filtered['region'] == region]

    postday_region = df_filtered.groupby(['post_day', 'region']).size().reset_index(name='count')
    if postday_region.empty:
        st.info("No data available for this selection.")
        return

    plt.figure(figsize=(12, 6))
    sns.barplot(data=postday_region, x="post_day", y="count", hue="region", palette="Set2")
    plt.title(f"Posts by Day and Region for Category: {category}, Region: {region}", fontsize=16)
    plt.ylabel("Count of Posts")
    plt.xlabel("Day of Week")
    plt.legend(title="Region", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    st.pyplot(plt)
    plt.clf()
