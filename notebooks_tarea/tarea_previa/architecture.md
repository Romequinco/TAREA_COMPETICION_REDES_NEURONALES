# Arquitectura del Sistema

## Visión general

El taller implementa un pipeline de forecasting financiero multivariante. Los datos fluyen desde yfinance, se transforman en ventanas deslizantes, se usan para entrenar y comparar 16 familias de redes neuronales, y finalmente se aplican a la construcción de carteras para 2025.

```
yfinance (datos crudos, 23 activos SP500, desde 1945)
    ↓ log-retornos
returns ∈ ℝ^(T × 23)   — T ≈ 16.200 días
    ↓ create_time_series_data(V_in, V_out)
X ∈ ℝ^(N × V_in × 23)    y ∈ ℝ^(N × 23)
    ↓ make_splits()
train (~72%) / val (~18%) / test (10%)  — orden cronológico, sin shuffle
    ↓
[01] Baselines      →  naive, lineal                       (32 combos)
[02] MLP            →  mlp_s                               (16 combos)
[03] Recurrentes    →  simple_rnn, gru, lstm,              (96 combos)
                       lstm_stack, bi_gru, lstm_drop
[04] Conv1D         →  conv_s                              (16 combos)
[05] Mixtos         →  conv_lstm_ln, conv_gru_bottleneck,  (96 combos)
                       conv_bilstm, conv2_lstm,
                       lstm_dense, conv_dense
    ↓
[06] Comparación global → 256 resultados, mejor MAE por (V_in, V_out)
    ↓
[07] Investigación  → StandardScaler · Rolling Z-score · FFD(d) · Feature Engineering
[08] Carteras 2025  → Buy&Hold · Cartera NN (long/short)
```

---

## Módulo compartido — `src/utils.py`

Importado por todos los notebooks con:
```python
sys.path.insert(0, os.path.join(os.getcwd(), '..', 'src'))
from utils import TICKERS, create_time_series_data, make_splits, ...
```

### Constantes globales
| Constante | Valor | Descripción |
|-----------|-------|-------------|
| `TICKERS` | lista 23 activos | Universo de activos SP500 |
| `INPUT_WINDOWS` | `[5, 10, 30, 90]` | Días de historia en X |
| `OUTPUT_WINDOWS` | `[1, 5, 30, 90]` | Días de horizonte en y |
| `RANDOM_SEED` | `42` | Semilla de partición (no cambiar) |
| `N_ASSETS` | `23` | Dimensión de features |

### Funciones principales

| Función | Firma | Descripción |
|---------|-------|-------------|
| `create_time_series_data` | `(data, V_in, V_out) → X, y` | Ventanas deslizantes; función del profesor, no modificar |
| `make_splits` | `(X, y, seed) → X_tr, X_v, X_ts, y_tr, y_v, y_ts` | Partición en dos pasos, shuffle=False |
| `eval_mae` | `(model, X, y) → float` | MAE medio sobre los 23 activos |
| `eval_mae_naive` | `(X, y) → float` | MAE del último valor conocido |
| `get_callbacks` | `(patience_lr) → list` | ReduceLROnPlateau + ModelCheckpoint |
| `restore_best_weights` | `(model)` | Restaura pesos del mejor epoch tras fit() |
| `compile_model` | `(model, lr) → model` | MAE loss + Adam; estándar para todos los modelos |
| `build_results_df` | `(results) → DataFrame` | Dict → MultiIndex (modelo, V_in, V_out) |
| `best_per_window` | `(df, metric) → DataFrame 4×4` | Mejor MAE por combinación de ventanas |
| `plot_history` | `(hist, title)` | Curva loss/val_loss por época |
| `plot_mae_matrix` | `(mat_df, title)` | Heatmap seaborn 4×4 |
| `plot_model_comparison` | `(df, V_in, V_out, metric)` | Barplot MAE por modelo |

---

## Formato de datos a través del pipeline

### Tensores
```
X : (N, V_in, 23)   — ventana de entrada (log-retornos)
y : (N, 23)         — promedio de cierres futuros (target)
```

### Partición cronológica
```
|──────────── train (~72%) ────────────|──── val (~18%) ────|── test (10%) ──|
                                        ↑ no shuffle en ningún paso
```

### Formato de resultados (dict estándar)
```python
results = {
    (nombre_modelo, V_in, V_out): {
        'train': float,   # MAE en train
        'val':   float,   # MAE en validación
        'test':  float,   # MAE en test
        'params': int     # nº de parámetros del modelo
    }, ...
}
```
Este dict se genera en cada notebook 01–05 y se agrega manualmente en `06_resultados.ipynb`.

---

## Familias de modelos

### Input shape por familia

| Familia | Input a Keras | Preprocesado |
|---------|--------------|--------------|
| Baselines (naive, lineal) | — / `(N, V_in*23)` | Flatten para lineal |
| MLP | `(N, V_in*23)` | `X.reshape(N, -1)` |
| Recurrentes (RNN, GRU, LSTM) | `(N, V_in, 23)` | Directo |
| Conv1D | `(N, V_in, 23)` | Directo; V_in ≥ kernel_size=3 |
| Mixtos (Conv + RNN) | `(N, V_in, 23)` | Directo |

### Todos los modelos activos (256 entrenamientos)

```python
# 01_baselines — sin Keras
naive:               y_pred = X[:, -1, :]                               # 16 combos
lineal:              LinearRegression().fit(X.reshape(N,-1), y)          # 16 combos

# 02_mlp — EPOCHS=50 (QUICK_MODE activo)
mlp_s:               Input(V*23) → Dense(64, relu, L2=1e-4) → Dense(23) # 16 combos

# 03_recurrentes — EPOCHS=300
simple_rnn:          Input(V,23) → SimpleRNN(32) → Dense(23)            # 16 combos
gru:                 Input(V,23) → GRU(32) → Dense(23)                  # 16 combos
lstm:                Input(V,23) → LSTM(32) → Dense(23)                 # 16 combos
lstm_stack:          Input(V,23) → LSTM(64, return_seq) → LSTM(32) → Dense(23)  # 16 combos
bi_gru:              Input(V,23) → Bidirectional(GRU(32)) → Dense(23)   # 16 combos
lstm_drop:           Input(V,23) → LSTM(64, dropout=0.2) → Dense(23)    # 16 combos

# 04_convolucionales — EPOCHS=700
conv_s:              Input(V,23) → Conv1D(64,k=3)×3 → Dense(64) → Dense(23) # 16 combos

# 05_mixtos — EPOCHS=500
conv_lstm_ln:        Input(V,23) → Conv1D(32,k=3) → LSTM(32,drop=0.1) → Dense(23)         # 16 combos
conv_gru_bottleneck: Input(V,23) → Conv1D(64,k=3) → Conv1D(16,k=1) → GRU(32,drop=0.1) → Dense(23) # 16 combos
conv_bilstm:         Input(V,23) → Conv1D(64,k=3) → SpatialDrop(0.15) → BiLSTM(32,drop=0.15) → Dense(23) # 16 combos
conv2_lstm:          Input(V,23) → Conv1D(64,k=3) → SpatDrop(0.12) → Conv1D(32,k=3) → SpatDrop(0.12) → LSTM(64,drop=0.1) → Dense(23) # 16 combos
lstm_dense:          Input(V,23) → LSTM(64) → Dense(64, relu) → Dense(23)                   # 16 combos
conv_dense:          Input(V,23) → Conv1D(64,k=3) → GAP → Dense(64, relu) → Dense(23)       # 16 combos
```

**Total: 16 modelos × 16 combinaciones = 256 entrenamientos**

### EPOCHS por notebook

| Notebook | EPOCHS | Justificación |
|----------|--------|---------------|
| 02_mlp | 50 (QUICK_MODE=True) | MAE converge rápido; colapso confirmado en <50 épocas |
| 03_recurrentes | 300 | Estándar del taller |
| 04_convolucionales | 700 | Sin QUICK_MODE; curvas completas documentadas |
| 05_mixtos | 500 | Balance entre convergencia y tiempo de cómputo |

---

## Flujo de compilación y entrenamiento

```python
# Todos los modelos usan la misma configuración base
model = compile_model(Sequential([...]))   # MAE + Adam(lr=3e-4)
hist  = model.fit(X_tr, y_tr,
                  validation_data=(X_v, y_v),
                  epochs=EPOCHS,
                  batch_size=64,
                  callbacks=get_callbacks(),   # ReduceLR + ModelCheckpoint
                  verbose=0)
restore_best_weights(model)   # recupera el mejor epoch
```

Sin EarlyStopping: el modelo entrena todas las épocas para ver la curva completa. `ModelCheckpoint` guarda el mejor estado en disco temporal; `restore_best_weights(model)` lo recupera al finalizar. El bucle es idéntico en todos los notebooks 02–05.

**Excepción**: `mlp_s` usa `lr=1e-4` (no el default 3e-4 de `compile_model`) — ver D25.

---

## Notebooks de investigación y carteras

### `07_investigacion.ipynb` — Preprocesado avanzado

Evalúa 5 técnicas de preprocesado sobre V_in=30, comparadas contra el baseline crudo:

| Técnica | MAE test (V_out=1) | Δ vs crudo |
|---------|-------------------|-----------|
| Crudo (baseline) | 0.0123 | — |
| StandardScaler | 0.0128 | +4.1% |
| Rolling Z-score | 0.0126 | +2.4% |
| **FFD (d=0.2)** | **0.0112** | **−8.9%** |
| Feature Engineering | 0.0125 | +1.6% |

**Única mejora real: FFD(d=0.2) para V_out=1.** Para V_out≥5 el FFD empeora.

```
returns_raw → FFD(d=0.2) → entrenar LSTM → comparar MAE crudo vs preprocesado
           → barrido d ∈ [0.1, 0.5] → d=0.2 óptimo
           → extensiones: todos los V_out, combinación FFD+Features, Feature Engineering multi-rama
```

### `08_carteras.ipynb` — Portfolio 2025

```
Datos históricos hasta 2024
    → entrenar mlp_s (V_in=10, V_out=90) sobre todo el histórico
    → predecir retornos para los próximos 90 días (última ventana de 2024)
    → construir pesos fijos:
        pesos_bh = [1/23, ..., 1/23]
        pesos_nn = y_pred / sum(|y_pred|)   # long/short
    → descargar datos 2025 → calcular retornos de cartera
    → métricas: retorno total, anual, volatilidad, Sharpe, Sortino, MaxDD
```

Todos los activos predicen retorno positivo (colapso a la media, que es positiva en el histórico largo), por lo que la cartera NN es long-only en la práctica.

---

## Hallazgo central: colapso universal al predictor de la media

Todos los modelos convergen al mismo MAE, equivalente a predecir siempre la media de los retornos de entrenamiento (≈ 0):

- `std(pred) / std(y_train) ≈ 0.06–0.10` en todos los modelos
- Diferencias entre arquitecturas: Δtest < 0.0001 en 15/16 combinaciones
- V_in no tiene impacto — MLP con 5 días = MLP con 90 días

**Causa**: log-retornos ≈ ruido blanco (EMH forma débil). El estimador que minimiza MAE es la media. Sin señal en el input, predecir la media es la respuesta óptima.

---

## Convenciones de código

- Cada notebook es autocontenido: importa desde `src/utils.py` y descarga sus propios datos.
- `QUICK_MODE = True` reduce `EPOCHS` a 50 en notebooks 02–05 (solo activo en 02 en producción).
- Los dicts `results` de cada notebook se agregan manualmente en `06_resultados.ipynb`.
- Los marcadores `# [EXTENDER]` indican extensiones opcionales no incluidas en el recuento de 256.
