import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Configurando o estilo do gráfico
sns.set(style='whitegrid', palette='pastel', color_codes=True)

# Configuração
np.random.seed(0)
n_experiments = [10, 100, 1000, 5000, 10000, 50000]
n_trials = 100000
expected_value = 0.5  # para uma moeda justa

# Fazendo as "jogadas"
for n, color in zip(n_experiments, sns.color_palette()):
    trials = np.random.randint(2, size=(n_trials, n))  # 0 para "coroa", 1 para "cara"
    mean_outcomes = trials.mean(axis=1)
    sns.histplot(mean_outcomes, bins=11, color=color, alpha=0.75, label=f'n={n}')

# Exibindo a expectativa
#plt.axvline(expected_value, color='red', linestyle='dashed', linewidth=2, label='Expectativa')

plt.legend()
plt.xlabel('Média dos resultados', fontsize=13)
plt.ylabel('Probabilidade', fontsize=13)
plt.title('Demonstração do TGN com lançamentos de moeda', fontsize=15)
plt.show()
