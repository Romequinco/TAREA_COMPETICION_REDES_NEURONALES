# Resumen de la Tarea Previa — Redes Neuronales para Forecasting Financiero

Referencia estática de la tarea B3-T4/T5/T6 (entrega 21-05-2026).
Servir como contexto rápido para el hackathon; no modificar.

---

## El problema

**Regresión multivariante**: predecir `y ∈ ℝ^23` (promedio de retornos logarítmicos
futuros de 23 activos SP500) a partir de `X ∈ ℝ^(V_in × 23)` (retornos pasados).

- **Activos**: AEP BA CAT CNP CVX DIS DTE ED GD GE HON HPQ IBM IP JNJ KO KR MMM MO MRK MSI PG XOM
- **Datos**: yfinance desde 1945, ~16 200 días de trading
- **Formato X/y**: `create_time_series_data` (ventanas deslizantes, función del profesor)
- **Métrica**: MAE medio sobre los 23 activos

---

## Datos y preprocesado

```
precios (cierre diario, 23 activos)
  ↓ np.log(precios).diff().dropna()
returns ∈ ℝ^(T × 23)    T ≈ 16 000 días
  ↓ create_time_series_data(V_in, V_out)
X ∈ ℝ^(N × V_in × 23)   y ∈ ℝ^(N × 23)
  ↓ make_splits()   → shuffle=False siempre
72 % train / 18 % val / 10 % test  (cronológico)
```

**Regla crítica**: shuffle=False siempre. Mezclar implica data leakage masivo.

---

## Modelos probados (256 entrenamientos)

| Familia | Modelos | Params | EPOCHS |
|---------|---------|--------|--------|
| Baselines | naive, lineal | 0 / 5K | — |
| MLP | mlp_s: Flatten→Dense(64,L2)→Dense(23) | 8K–134K | 50 |
| Recurrentes | simple_rnn, gru, lstm, lstm_stack, bi_gru, lstm_drop | 2K–27K | 300 |
| Conv1D | conv_s: Conv1D(64)×3→Dense(64)→Dense(23) | 51K | 700 |
| Mixtos | conv_lstm_ln, conv_gru_bottleneck, conv_bilstm, conv2_lstm, lstm_dense, conv_dense | 10K–37K | 500 |

**Ventanas**: INPUT_WINDOWS=[5,10,30,90] × OUTPUT_WINDOWS=[1,5,30,90] → 16 combinaciones por modelo.

---

## Resultado central: convergencia a zona estrecha

Todos los modelos convergen al mismo MAE, equivalente a la regresión lineal:

| V_out | MAE todas las NN | vs Naive | vs Lineal |
|-------|-----------------|----------|-----------|
| 1d    | ≈ 0.0123        | −31 %    | ≈ 0 %     |
| 5d    | ≈ 0.0056        | −59 %    | ≈ 0 %     |
| 30d   | ≈ 0.0023        | −82 %    | ≈ 0 %     |
| 90d   | ≈ 0.0013        | −89 %    | ≈ 0 %     |

Diferencias entre arquitecturas: **Δtest < 0.0001 en 15/16 combinaciones**.
V_in no tiene impacto — MLP con 5 días = MLP con 90 días.

**Por qué**: log-retornos ≈ ruido blanco (EMH). El estimador que minimiza MAE
es la media. Sin señal real en el input, predecir la media es la respuesta óptima.

---

## La única mejora real: FFD(d=0.2)

Probada en notebook `07_investigacion.ipynb`. Barrido d ∈ [0.1, 1.0]:

| d   | MAE test (V_out=1) | Δ vs crudo |
|-----|-------------------|------------|
| 0.1 | 0.0122            | −0.8 %     |
| **0.2** | **0.0112**    | **−8.9 %** |
| 0.4 | 0.0114            | −7.3 %     |
| 1.0 | 0.0124            | +0.8 %     |

**d=0.2 óptimo para V_out=1. Para V_out=30 empeora +45 %.**

El log-precio con d=0.2 retiene memoria a largo plazo que los retornos puros
(d=1) descartan. La mejora es exclusiva del horizonte corto.

---

## Preprocesados que empeoran

| Técnica | V_out=1 | V_out=30 |
|---------|---------|---------|
| StandardScaler | +4.1 % ✗ | +8.3 % ✗ |
| Rolling Z-score | +2.4 % ✗ | +8.3 % ✗ |
| Feature Engineering (vol+momentum+corr) | +1.6 % ✗ | +8.3 % ✗ |

Normalizar elimina la escala absoluta de los retornos, que contiene información
sobre el régimen de volatilidad. Peor modelo, no mejor.

---

## Decisiones de diseño importantes

| Decisión | Motivo |
|----------|--------|
| **MAE como loss** (no MSE) | Lo exige el enunciado; más robusto a outliers financieros |
| **Adam lr=3e-4** (default) | Estándar del material teórico; excepción: lr=1e-4 para MLP |
| **Sin EarlyStopping** | El profesor lo prohíbe; usar ReduceLROnPlateau + ModelCheckpoint |
| **batch_size=64** | Balance velocidad/calidad del gradiente; CPU-friendly |
| **L2=1e-4 en Dense(64) del MLP** | Mejora val_min en 15/16 combinaciones |
| **No normalizar** | StandardScaler y RollingZ empeoran consistentemente |
| **dropout en mixtos** | Solo con capas grandes (22K+ params) para evitar sobreajuste |

---

## Errores a no repetir

- **Shuffle=True** en el split → data leakage, MAE test artificialmente bajo
- **StandardScaler antes de crear ventanas** → escala global pierde info de régimen
- **Más de 3 Conv1D apiladas sin padding='same'** → falla con V_in=5 (kernel>input)
- **EarlyStopping** → el profesor lo prohíbe; interrumpe antes de convergencia real
- **Comparar MAE de un solo entrenamiento** sin semilla fija → diferencias dentro del ruido

---

## Modelo de referencia recomendado

Para el hackathon, punto de partida sugerido:

```python
# 1. LSTM + ensemble de semillas (palanca principal)
train_ensemble('lstm', ..., n_seeds=5, epochs=300, lr=3e-4)

# 2. Si V_OUT = 1: añadir FFD primero
X_src, y_src = load_data('data/precios.csv', ffd_d=0.2)
train_ensemble('lstm', ..., n_seeds=5, epochs=300)
```
