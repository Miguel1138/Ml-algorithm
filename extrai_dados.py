import pandas as pd

df = pd.read_csv("Titanic-Dataset.csv")

# Etração do título
df["Title"] = df["Name"].str.extract(r",\s*([^\.]+)\.")
df["Title"] = df["Title"].replace(
    {"Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs"}
)

raros = df["Title"].value_counts()[lambda s: s < 10].index
df["Title"] = df["Title"].replace(raros, "Raros")

# Flags e agrupando familiares
df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
df["IsAlone"] = (df["FamilySize"] == 1).astype(int)
df["HasCabin"] = df["Cabin"].notna().astype(int)


# Aplica o valor da média para pvalores nulos.
df["Age"] = df["Age"].fillna(df.groupby("Title")["Age"].transform("median"))

# Aplica a moda para valores nulos de embarque para não interferir na média.
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

cols = ["Pclass", "Sex", "Age", "Fare", "Embarked",
        "Title", "FamilySize", "IsAlone", "HasCabin"]
x = pd.get_dummies(df[cols], drop_first=True)
y = df["Survived"]

