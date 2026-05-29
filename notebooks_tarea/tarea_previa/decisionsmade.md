# Decisiones de Diseño

Registro de todas las decisiones tomadas durante el diseño e implementación del taller. Cada entrada incluye la opción elegida y el razonamiento.

---

## D1 — Función de pérdida

**Decisión**: MAE (`mean_absolute_error`)

El enunciado del taller exige MAE como métrica de evaluación. Los notebooks del profesor usan MSE por simplicidad didáctica, pero el taller es explícito en requerir MAE. MAE es además más robusto a outliers, lo cual es relevante para retornos financieros con colas pesadas.

---

## D2 — Optimizador y learning rate

**Decisión**: `Adam(learning_rate=3e-4)`

Adam es la primera opción recomendada en el material teórico (`training-nn-2026.md`). `lr=3e-4` es el valor estándar para Adam en problemas de regresión con este rango de valores de target. Mismo valor que usan los notebooks del profesor. Excepción: `mlp_s` usa `lr=1e-4` (ver D25).

---

## D3 — Número máximo de épocas

**Decisión**: `EPOCHS` variable por notebook; sin EarlyStopping

El profesor indicó que EarlyStopping oculta el comportamiento real del entrenamiento y hace las curvas incomparables entre modelos.

| Notebook | EPOCHS | Motivo |
|----------|--------|--------|
| 02_mlp | 50 (QUICK_MODE=True) | Colapso confirmado en <50 épocas; QUICK_MODE no se desactivó en producción |
| 03_recurrentes | 300 | Estándar del taller |
| 04_convolucionales | 700 | Curvas completas sin QUICK_MODE |
| 05_mixtos | 500 | Balance tiempo/convergencia |

`QUICK_MODE = True` reduce a 50 épocas para pruebas rápidas (flag al inicio de cada notebook).

---

## D4 — Batch size

**Decisión**: `BATCH_SIZE = 64`

Balance entre velocidad de entrenamiento y calidad del gradiente. 64 es CPU-friendly y estándar para este tipo de problemas.

---

## D5 — Callbacks

**Decisión**: `ReduceLROnPlateau(patience=5, factor=0.5)` + `ModelCheckpoint(save_best_only=True)`

- **EarlyStopping eliminado** por indicación del profesor: oculta el comportamiento real del entrenamiento y hace las curvas incomparables entre modelos.
- **ReduceLROnPlateau**: reduce el LR a la mitad si no hay mejora en 5 épocas, permitiendo salir de mínimos locales planos sin cortar el entrenamiento.
- **ModelCheckpoint**: guarda el mejor estado (menor `val_loss`) en una ruta temporal durante el entrenamiento. Llamar `restore_best_weights(model)` tras `model.fit()` para recuperarlo.

---

## D6 — Guardado de modelos

**Decisión**: `ModelCheckpoint` con ruta temporal durante el bucle; sin persistencia entre notebooks

El checkpoint temporal (`tempfile.mktemp(suffix='.keras')`) se sobreescribe en cada entrenamiento — solo persiste el mejor epoch del modelo actual. Evita acumular 256 archivos `.keras` durante el bucle.

---

## D7 — Partición de datos

**Decisión**: Dos pasos con `shuffle=False`; validación ampliada al 20% del train_full

```python
# Paso 1: 90% train_full / 10% test
X_tr_full, X_ts, y_tr_full, y_ts = train_test_split(
    X, y, test_size=0.10, shuffle=False, random_state=42)
# Paso 2: 80% train / 20% val del train_full
X_tr, X_v, y_tr, y_v = train_test_split(
    X_tr_full, y_tr_full, test_size=0.20, shuffle=False, random_state=42)
```

`shuffle=False` es obligatorio para series temporales financieras. Esta partición produce ~72% train / ~18% val / 10% test. El 20% de val (vs. 5% original del profesor) da señal más robusta para `ReduceLROnPlateau` y `ModelCheckpoint` al haber eliminado EarlyStopping.

---

## D8 — Función `create_time_series_data`

**Decisión**: Usar exactamente la del profesor, sin modificaciones

La función genera `X:(N, V_in, N_assets)` e `y:(N, N_assets)` como promedio de los `V_out` pasos futuros. No se modifica para garantizar reproducibilidad y compatibilidad con los resultados del profesor.

---

## D9 — Baselines

**Decisión**: Naive (último valor conocido) + Regresión Lineal (sklearn)

- **Naive**: `y_pred = X[:, -1, :]` — predice el último retorno observado.
- **Lineal**: `LinearRegression` de sklearn sobre X aplanado. Baseline explícito del profesor.

---

## D10 — Modelos activos y total de entrenamientos

**Decisión**: 14 modelos NN × 16 combinaciones = 224 entrenamientos de red neuronal (256 totales incluyendo baselines)

| Notebook | Modelos activos | Entrenamientos |
|----------|----------------|----------------|
| 01_baselines | naive, lineal | 32 |
| 02_mlp | mlp_s | 16 |
| 03_recurrentes | simple_rnn, gru, lstm, lstm_stack, bi_gru, lstm_drop | 96 |
| 04_convolucionales | conv_s | 16 |
| 05_mixtos | conv_lstm_ln, conv_gru_bottleneck, conv_bilstm, conv2_lstm, lstm_dense, conv_dense | 96 |
| **Total** | **16 modelos** | **256** |

Supera ampliamente el mínimo de 64 entrenamientos requerido.

---

## D11 — Input shape para MLP

**Decisión**: Flatten a `(N, V_in * 23)` con `X.reshape(N, -1)`

MLP no admite entrada 3D nativa. El flatten es la transformación estándar y la que usan los notebooks del profesor. Como consecuencia, el MLP pierde completamente la estructura temporal — V_in no aporta información más allá del primer día.

---

## D12 — Conv1D kernel size

**Decisión**: `kernel_size=3`

Kernel mínimo para capturar patrones locales de al menos 3 días. Compatible con `V_in=5` (el mínimo en `INPUT_WINDOWS`). `padding='same'` en modelos mixtos para preservar la longitud de secuencia en V_in pequeños.

---

## D13 — Rebalanceo de carteras

**Decisión**: Pesos **fijos** durante todo 2025; rebalanceo mensual marcado como `[EXTENDER]`

Pesos fijos simplifican la comparación y son el comportamiento estándar descrito en el enunciado. El rebalanceo mensual (~21 días de trading) está documentado en el código como extensión opcional.

---

## D14 — Construcción de pesos de la cartera NN

**Decisión**: Long/Short proporcional a retorno predicho, normalizado por suma de valores absolutos

```python
pesos_nn = y_pred / np.sum(np.abs(y_pred))
```

En la práctica, dado el colapso al predictor de la media positiva, todos los activos tienen predicción positiva y la cartera es long-only.

---

## D15 — Preprocesado en la competición

**Decisión**: Solo log-retornos; sin normalización adicional en notebooks 01–05

Los log-retornos ya tienen media ≈ 0 y están en la misma escala para todos los activos. La normalización avanzada se reserva para `07_investigacion.ipynb`.

---

## D16 — Preprocesado en la investigación

**Decisión**: Implementar y comparar 5 técnicas; FFD(d=0.2) es la única con mejora real

Técnicas implementadas en `07_investigacion.ipynb` (todas sobre V_in=30):

| Técnica | Δ MAE (V_out=1) | Activa |
|---------|----------------|--------|
| StandardScaler | +4.1% | Sí |
| Rolling Z-score | +2.4% | Sí |
| FFD (d=0.2) | −8.9% | Sí |
| Feature Engineering (vol, momentum) | +1.6% | Sí |

FFD es la única mejora confirmada, y solo para V_out=1 día. Para V_out≥5 empeora (añade ruido de largo plazo donde el modelo necesita estacionariedad).

---

## D17 — Visualizaciones

**Decisión**: `matplotlib` para curvas de convergencia; `seaborn` para heatmaps 4×4

`seaborn.heatmap` genera los heatmaps de MAE en una línea con anotaciones automáticas. No se introducen dependencias adicionales.

---

## D18 — Agregación de resultados en `06_resultados.ipynb`

**Decisión**: Pegar los dicts `results` manualmente; no hay ejecución automática entre notebooks

Cada notebook es autocontenido (descarga datos, entrena, evalúa). No existe persistencia automática entre notebooks para mantener el código simple y compatible con Colab. Los valores se copian de los outputs de ejecución.

---

## D19 — Número de activos

**Decisión**: 23 activos fijos (los que tienen datos completos desde 1945)

`precios.dropna(axis=1, inplace=True)` elimina automáticamente activos sin datos históricos completos.

---

## D20 — Período de evaluación de carteras

**Decisión**: Año 2025 completo (`start='2025-01-01'` sin `end`)

El enunciado especifica comparar rendimientos para 2025. Descargar sin fecha de fin captura todos los datos disponibles hasta la fecha de ejecución del notebook.

---

## D21 — Métricas de carteras

**Decisión**: Retorno total, retorno anual, volatilidad anual, Sharpe, Sortino, Max Drawdown

`TRADING_DAYS = 252` para anualización. El Sortino usa solo retornos negativos en el denominador, siendo más informativo que el Sharpe en distribuciones asimétricas.

---

## D22 — Estructura de archivos

**Decisión**: `src/utils.py` compartido + 9 notebooks independientes

Centralizar funciones en `utils.py` evita duplicación. Cada notebook sigue siendo autocontenido para facilitar la ejecución independiente en Colab.

---

## D23 — Marcadores de extensión

**Decisión**: Usar `# [EXTENDER]` para código comentado ampliable

Permite ir más allá de 256 modelos sin cambiar la estructura. Cada línea comentada es una extensión lista para activarse.

---

## D24 — `QUICK_MODE`

**Decisión**: Flag booleano al inicio de cada notebook, `EPOCHS = 50` cuando activo

Permite probar que todo el flujo funciona sin esperar las 3–4 horas de cada entrenamiento completo.

---

## D25 — Learning rate de `mlp_s`: 3e-4 → 1e-4

**Decisión**: `Adam(learning_rate=1e-4)` en `build_mlp` del notebook 02 (solo MLP)

El LR global de `utils.py` (`compile_model` default = 3e-4) no se modifica para no afectar a LSTM, GRU y Conv.

**Evidencia (Evidencia 3, 300 épocas, 16 combinaciones):**
- Baseline LR=3e-4: best epoch 3–28 en 15/16 combinaciones (convergencia demasiado rápida)
- Variante LR=1e-4: best epoch 8–300 — 14/16 celdas mejoran best_epoch
- Δval_min máximo: +0.00009 (umbral de rechazo: 0.0005) → sin coste en calidad

---

## D26 — L2=1e-4 en Dense(64) de mlp_s

**Decisión**: `kernel_regularizer=l2(1e-4)` en la capa Dense(64) de `build_mlp`

val_min mejora en 15/16 combinaciones; único caso con Δval > 0: (90,1) con +0.00004 (< umbral 0.0005).

---

## D27 — Regularización en modelos mixtos (05_mixtos, Sección B)

**Decisión**: `dropout` en capas recurrentes + `SpatialDropout1D` en capas convolucionales; ajuste asimétrico por modelo

Las arquitecturas `conv_bilstm` (22K params) y `conv2_lstm` (37K params) tenían sobreajuste severo sin regularización. Se ajustó en 4 iteraciones:

| Iteración | dropout LSTM | SpatialDropout1D | Resultado |
|-----------|-------------|-----------------|-----------|
| 1 — Sin regularización | — | — | Divergencia severa |
| 2 — Dropout LSTM | 0.1 | — | `conv_bilstm` OK, `conv2_lstm` parcial |
| 3 — Dropout + SpatialDrop | 0.1 | 0.1 | Curvas convergen |
| **4 — Ajuste fino (final)** | **0.15 / 0.1** | **0.15 / 0.12** | **Curvas limpias** |

`SpatialDropout1D` es preferible al `Dropout` estándar para salidas convolucionales porque apaga canales completos (alta correlación interna) en lugar de neuronas individuales. El ajuste es asimétrico: `conv_bilstm` (una sola Conv1D) necesita más dropout que `conv2_lstm` (dos Conv1D apiladas cuyo efecto se compone).

**Observación clave**: las 4 iteraciones produjeron exactamente el mismo MAE en test. La regularización mejora las curvas pero no el resultado — el techo es del problema (EMH), no del modelo.

---

## D28 — No intentar superar el techo del MAE con más arquitecturas

**Decisión**: Priorizar diversidad de familias (6 recurrentes, 6 mixtos) sobre profundidad de ajuste de cada arquitectura

Tras confirmar experimentalmente que todos los modelos convergen al mismo MAE (~0.0123 para V_out=1), el valor añadido de ajustar cada arquitectura individualmente es marginal. Se priorizó explorar más familias y dedicar tiempo a la investigación (notebook 07) donde FFD(d=0.2) sí produce una mejora real (−8.9%).

---

## D29 — Arquitecturas de 03_recurrentes: 6 modelos en lugar de los 2 mínimos

**Decisión**: simple_rnn, gru, lstm, lstm_stack, bi_gru, lstm_drop — todos con unidades reducidas (32u base)

Las unidades se redujeron de 64 a 32 para poder incluir 6 modelos en el mismo tiempo de cómputo. Con 32u el MAE es idéntico (colapso al predictor de la media), por lo que la capacidad adicional no aporta. `lstm_drop` (64u, dropout=0.2) es la excepción — tiene más parámetros para que el dropout tenga efecto regularizador visible.
