# Aprendendo Machine Learning na prática

Repositório de estudo para aprender **algoritmos de Machine Learning** e as principais bibliotecas Python da área (`pandas`, `numpy`, `scikit-learn`, `PyTorch`, `matplotlib`).

O objetivo **não é obter a melhor performance**, e sim entender como cada algoritmo funciona, quando usá-lo e como avaliá-lo corretamente. Por isso, o mesmo problema é resolvido com algoritmos de complexidade crescente e os resultados são comparados.

## Problema usado

**Titanic** (`Titanic-Dataset.csv`, 891 passageiros): prever se o passageiro sobreviveu (`Survived`) a partir de classe, sexo, idade, família, tarifa e porto de embarque. É uma tarefa de **classificação binária** com dados tabulares, pequena o bastante para treinar em segundos e rica o bastante para praticar limpeza de dados e engenharia de features.

## O que é praticado

1. Exploração de dados com `pandas`
2. Limpeza (valores nulos) e *feature engineering* (título do nome, tamanho da família, flags)
3. Divisão treino/validação/teste e validação cruzada
4. Três algoritmos de classificação, do mais simples ao mais flexível
5. Avaliação com métricas além da acurácia
6. Diagnóstico de overfitting

## Algoritmos

### 1. Regressão Logística (baseline)

- **Como funciona:** calcula uma soma ponderada das features e a passa por uma função sigmoide, que devolve uma probabilidade entre 0 e 1. Se for maior que 0,5, prevê "sobreviveu".
- **Por que usar:** é simples, rápida e interpretável (cada coeficiente indica o peso de uma feature). Serve de **linha de base**: qualquer modelo mais complexo precisa superá-la para se justificar.
- **Limitação:** só aprende fronteiras lineares; não captura interações entre variáveis sem que você as crie manualmente.
- **Requer:** features numéricas escalonadas (`StandardScaler`).
- **Implementação:** `sklearn.linear_model.LogisticRegression`

### 2. Random Forest

- **Como funciona:** treina muitas árvores de decisão, cada uma em uma amostra aleatória dos dados e das features (*bagging*), e combina as previsões por votação. A diversidade entre as árvores reduz o overfitting de uma árvore isolada.
- **Por que usar:** captura relações não lineares e interações automaticamente, exige pouco pré-processamento e fornece a **importância das features**.
- **Limitação:** menos interpretável que a regressão logística e mais pesado em memória.
- **Hiperparâmetros principais:** `n_estimators`, `max_depth`, `min_samples_leaf`.
- **Implementação:** `sklearn.ensemble.RandomForestClassifier`

### 3. Rede Neural (MLP) em PyTorch

- **Como funciona:** o *Multi-Layer Perceptron* empilha camadas lineares com funções de ativação não lineares (ReLU). Os pesos são ajustados por **gradiente descendente** (otimizador Adam), minimizando a *binary cross-entropy* via retropropagação (`backward()`).
- **Por que usar:** é a base do deep learning. Treiná-la "na mão" ensina o ciclo completo: *forward → loss → backward → step*.
- **Técnicas de regularização usadas:** `Dropout`, `weight_decay` e *early stopping* (parar quando a loss de validação deixa de melhorar).
- **Limitação:** em dados tabulares pequenos, costuma **não superar** modelos de árvores; aqui o valor está no aprendizado do pipeline.
- **Implementação:** `torch.nn.Sequential`, `BCEWithLogitsLoss`, `torch.optim.Adam`

## Técnicas de apoio

| Técnica | Para que serve |
| --- | --- |
| `StandardScaler` | Coloca as features na mesma escala (média 0, desvio 1); ajustado **só no treino** |
| One-hot encoding (`pd.get_dummies`) | Converte categorias em colunas numéricas |
| Imputação por mediana/moda | Preenche valores nulos |
| `train_test_split(stratify=y)` | Divide os dados mantendo a proporção das classes |
| Validação cruzada (`StratifiedKFold`) | Estima a performance de forma mais estável que um único split |
| `Pipeline` | Encadeia pré-processamento e modelo, evitando vazamento de dados |

## Métricas de avaliação

| Métrica | Pergunta que responde |
| --- | --- |
| Acurácia | Qual percentual de previsões está certo? (engana com classes desbalanceadas) |
| Precision | Dos que previ como positivos, quantos realmente eram? |
| Recall | Dos positivos reais, quantos encontrei? |
| F1 | Equilíbrio entre precision e recall |
| ROC-AUC | Quão bem o modelo separa as classes em todos os limiares? |
| Matriz de confusão | Onde o modelo erra: falsos positivos ou falsos negativos? |

O conjunto de **teste é usado uma única vez**, ao final. Comparações e ajustes usam validação cruzada ou o conjunto de validação.

## Como executar

```bash
git clone <url-do-repositorio>
cd <nome-do-repositorio>

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install pandas numpy matplotlib seaborn scikit-learn torch

python titanic.py                  # ou abra o notebook, se preferir
```

Coloque o arquivo `Titanic-Dataset.csv` na raiz do projeto.

## Estrutura sugerida

```text
.
├── README.md
├── Titanic-Dataset.csv
├── requirements.txt
└── titanic.py        # exploração, limpeza, modelos e avaliação
```

À medida que o projeto crescer, separe em arquivos por etapa (limpeza, regressão logística, random forest, rede neural).

## Limitações conhecidas

- O dataset é pequeno (891 linhas), então os resultados variam conforme o `random_state` e o split; use a validação cruzada para comparar modelos.
- A imputação de `Age` está calculada no dataset inteiro, o que causa um vazamento leve de dados. Em projetos reais, ajuste imputações apenas no treino, dentro de um `Pipeline`.

## Próximos passos

- [ ] Encapsular o pré-processamento em `Pipeline` + `ColumnTransformer`
- [ ] Ajustar hiperparâmetros com `GridSearchCV` ou `Optuna`
- [ ] Testar gradient boosting (`XGBoost`/`LightGBM`) e explicar previsões com `SHAP`
- [ ] Repetir o fluxo em outro dataset (por exemplo, dados do DATASUS)
- [ ] Usar `nn.Embedding` para variáveis categóricas na rede neural

## Licença

Uso livre para fins educacionais.
