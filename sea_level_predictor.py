import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

def draw_plot():
    # Leitura dos dados
    df = pd.read_csv("epa-sea-level.csv")

    # Criar figuras
    fig, ax = plt.subplots(figsize=(10, 6))

    # Scatter plot dos dados históricos
    ax.scatter(df["Year"], df["CSIRO Adjusted Sea Level"], label="Observations")

    # Regressão linear 1: usando todos os dados históricos
    # X = Year, y = sea level
    X1 = sm.add_constant(df["Year"])  # adiciona intercepto
    model1 = sm.OLS(df["CSIRO Adjusted Sea Level"], X1).fit()
    # Prever até ano 2050
    y1_pred = model1.predict(sm.add_constant(pd.Series(range(df["Year"].min(), 2051))))
    ax.plot(
        range(df["Year"].min(), 2051),
        y1_pred,
        color="red",
        label="Fit all data"
    )

    # Regressão linear 2: usando dados a partir de 2000
    df_recent = df[df["Year"] >= 2000]
    X2 = sm.add_constant(df_recent["Year"])
    model2 = sm.OLS(df_recent["CSIRO Adjusted Sea Level"], X2).fit()
    y2_pred = model2.predict(sm.add_constant(pd.Series(range(2000, 2051))))
    ax.plot(
        range(2000, 2051),
        y2_pred,
        color="green",
        label="Fit from 2000"
    )

    # Labels / título / legenda
    ax.set_xlabel("Year")
    ax.set_ylabel("Sea Level (inches)")
    ax.set_title("Rise in Sea Level")
    ax.legend()

    # Salvar figura como arquivo (se exigido pelo boilerplate)
    fig.savefig("sea_level_plot.png")
    return fig