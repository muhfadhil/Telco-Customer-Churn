import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# pyplot settings
plt.style.use("ggplot")
plt.rcParams["font.size"] = 10
plt.rcParams["axes.labelsize"] = 10
plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams["axes.titlesize"] = 10
plt.rcParams["xtick.labelsize"] = 8
plt.rcParams["ytick.labelsize"] = 8
plt.rcParams["legend.fontsize"] = 8
plt.rcParams["figure.titlesize"] = 12
plt.rcParams["xtick.color"] = "black"
plt.rcParams["ytick.color"] = "black"
plt.rcParams["figure.dpi"] = 300

# Load the data
df = pd.read_pickle("../../data/interim/01_custid_dropped.pkl")

# Function for saving image
IMAGES_PATH = "../../reports/figures"


def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = f"{IMAGES_PATH}/{fig_id}.{fig_extension}"
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)


# Univariate

## Numeric columns
num_cols = df.select_dtypes(include=["int64", "float64"]).columns

### Distribution
for col in num_cols:
    # Statistic descriptive
    average = df[col].mean()
    median = df[col].median()
    mode = df[col].mode()[0]
    std = df[col].std()

    # Make subplot for histogram
    fig, ax = plt.subplots(figsize=(6, 3))
    sns.histplot(data=df, x=col, kde=True)
    plt.axvline(average, color="r", linestyle="solid", linewidth=3, label="Mean")
    plt.axvline(median, color="y", linestyle="dotted", linewidth=3, label="Median")
    plt.axvline(mode, color="b", linestyle="dashed", linewidth=3, label="Mode")
    plt.legend(
        loc="upper center",
        ncol=3,
        bbox_to_anchor=(0.5, 1.17),
        fancybox=True,
        shadow=True,
    )
    save_fig(f"Dist_{col}")
    plt.show()

### Boxplot
plt.figure(figsize=(20, 5))
sns.boxplot(data=df, orient="y")
save_fig("boxplot")
plt.show()

## Categorical columns
cat_cols = df.select_dtypes(include=["object"]).columns

### Barplot
for col in cat_cols:
    fig, ax = plt.subplots(figsize=(6, 3))
    df[col].value_counts().plot(kind="bar")
    plt.ylabel("number of customers")
    plt.xticks(rotation=6)
    for p in ax.patches:
        ax.annotate(
            f"{p.get_height()}",
            (p.get_x() + p.get_width() / 2.0, p.get_height() / 4),
            ha="center",
            va="baseline",
            fontsize=8,
            color="white",
            xytext=(0, 5),
            textcoords="offset points",
        )
    save_fig(f"Dist_{col}")
    plt.show()

# Bivariate
# plt.figure(figsize=(10,10))
sns.pairplot(data=df, hue="Churn")
plt.show()

## Categorical data
for col in cat_cols:
    plt.figure(figsize=(7, 4))
    sns.countplot(x=df[col], hue=df["Churn"])
    plt.legend(
        loc="upper center",
        ncol=2,
        bbox_to_anchor=(0.5, 1.17),
        fancybox=True,
        shadow=True,
    )
    plt.xticks(rotation=6)
    save_fig(f"Bivariate_{col}")
    plt.show()
