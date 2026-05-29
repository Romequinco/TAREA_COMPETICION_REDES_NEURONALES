# HACKATHON — Predicción autorregresiva de 6 índices financieros

## Contexto del problema (CAMBIÓ respecto a la tarea previa)

- **Tarea**: predecir 6 índices (`Index_A` … `Index_F`) de forma **autorregresiva**:
  252 días, **día a día**, usando la predicción de cada día como input del siguiente.
  Los **errores se acumulan en el rollout** — ese es el núcleo del problema.
- **Métrica**: **RMSE promedio sobre los 6 índices** (ya **NO** es MAE).
  RMSE penaliza el error al cuadrado → los índices volátiles (**A** alta vol., **F**
  extrema tipo cripto) **dominan** el error. Ahí se gana o se pierde la competición.
- **Aprobado**: RMSE < 75 000. El umbral **no informa de la magnitud** de los índices
  (no asumir escala hasta ver los datos el sábado).
- **Entrega**: CSV de **252 filas × 6 columnas**. Solo **6 entregas** permitidas; una
  idéntica a otra ya enviada **no cuenta**.
- **Stack**: Keras/TensorFlow (CPU) · pandas · numpy · Python 3.11. Modelos fundacionales
  (Chronos-2 / TimesFM) disponibles como alternativa zero-shot.

### Datos (se dan el sábado, múltiples ficheros)
| Fichero | Contenido |
|---------|-----------|
| `train_indices.csv` | cierres diarios de los 6 índices |
| `test_dates.csv` | las 252 fechas a predecir |
| `train_news.csv` / `test_news.csv` | titulares de noticias |
| `train_macro_factors.csv` / `test_macro_factors.csv` | oro, crudo, tipos |
| `train_network_metrics.csv` / `test_network_metrics.csv` | on-chain del `Index_F` |

### Pistas por índice (feature engineering regalado — un enfoque por índice)
| Índice | Pista | Estrategia probable |
|--------|-------|---------------------|
| **A** | Alta volatilidad / crecimiento | NN + ensemble; aquí pesa el RMSE |
| **B** | Steady-State defensivo | baseline plano probablemente lo clava |
| **C** | Energy-Pulse, ligado a macro | features de `macro_factors` |
| **D** | "The Ghost" — sigue una señal oculta en OTRO índice | `lagged_correlation` para cazar el lag |
| **E** | Global-ESG | NN estándar / baseline |
| **F** | Digital-Frontier, ligado a network | features de `network_metrics`; volátil, pesa mucho |

---

## ⚠️ WORKAROUND GPU — EJECUTAR SIEMPRE PRIMERO

RTX 5070 Ti (Blackwell) es **incompatible** con TensorFlow GPU.
Esta línea va **ANTES de cualquier import de TF/Keras** o el proceso se cuelga:

```python
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"   # CPU-only
# solo después:
import tensorflow as tf
```
Entorno de referencia (Daniel): Python 3.11.9 en `C:\venv_redes`.

---

## Estrategia: un enfoque INDEPENDIENTE por índice

No hay "un modelo para todo". Cada serie puede tener su mejor enfoque:

```
PASO 0 — Baselines primero (asegurar el aprobado)
  flat / drift / random_walk en backtest 252d → saber qué hay que batir por índice.

PASO 1 — Explotar las pistas
  D (Ghost): derivar de otro índice + lag (puede batir a cualquier NN).
  B (defensivo): baseline plano suele ganar — no compliques.
  C / F: añadir features auxiliares (macro / network).

PASO 2 — NN donde aporte (sobre todo A y F, que dominan el RMSE)
  LSTM por defecto en log-ret mode + ensemble de semillas.

PASO 3 — Backtest 252d de CADA enfoque y quedarse con el mejor POR ÍNDICE.
```

### Trabajar en LOG-RETORNOS, reconstruir en PRECIOS
No conocemos la escala de los índices. Para que las NN entrenen sea cual sea la magnitud,
**el pipeline opera en log-retornos** (escala ~0.01, agnóstica) y **reconstruye precios**
para medir el RMSE (la métrica oficial es sobre precios):

```
precio -> log-ret (precios_a_logret)  ->  modelo predice log-rets  ->
reconstruir: precio[t] = precio_inicial · exp(cumsum(log_rets))  (logret_a_precios)
```

`precio_inicial` = **último precio real conocido**. En el rollout **nunca** se usa un
precio real del futuro: solo ese anclaje + la cadena de predicciones propias. Esto lo
gestionan `predict_autoregressive` y `backtest_autoregressive` automáticamente
(`log_ret_mode=True`).

---

## Flujo del equipo (3 miembros)

```
ANTES DE EMPEZAR (5 min, todos):
  1. Soltar los CSV en data/
  2. Leer el enunciado y rellenar el diagnóstico (celda 3 del notebook)
  3. Acordar V_IN_SHARED viendo las series reales → editar UNA VEZ en utils.py

TRABAJO PARALELO (cada uno en su exp_*.ipynb):
  4. GPU workaround (celda 1, siempre)
  5. Carga + diagnóstico + baselines (celdas 3-4): asegurar el aprobado
  6. Pistas (celda 5) + un modelo por índice (celda 6) + ensemble (celda 7)
  7. Backtest 252d SIEMPRE → guardar mejor enfoque por índice (celda 8)

SÍNTESIS FINAL (COMPETICION.ipynb):
  8. Recoger el mejor enfoque de cada índice (venga del notebook que venga)
  9. Construir el vector 252×6, validar formato, reportar RMSE de backtest
  10. Generar el CSV. Si algo falla: sección de fallback manual.
```

### Reparto sugerido
- **Oscar** — infra + validación + entrega: `utils.py`, `backtest_autoregressive`,
  `generar_submission` / `validar_submission`, consolidación en `COMPETICION.ipynb`.
- **Miembro 2** — baselines + índices defensivos + Ghost: B (plano), E, y cazar el
  lag de D con `lagged_correlation`.
- **Miembro 3** — volátiles con datos auxiliares: A y F (los que dominan el RMSE),
  features de macro (C) y network (F), ensembles de semillas.

---

## Reglas de oro — NO negociables

1. **GPU workaround SIEMPRE primero** — `CUDA_VISIBLE_DEVICES=-1` antes de importar TF.
2. **Validar a 252d ANTES de subir** — `backtest_autoregressive` es el ÚNICO juez fiable.
   Un loss de entrenamiento bajo NO predice el RMSE del rollout. Nunca subir sin backtest.
3. **Solo 6 entregas** — cada una debe ser una mejora medida en backtest, no una corazonada.
   Una entrega idéntica a otra no cuenta. `validar_submission` antes de cada subida.
4. **El split de validación de 252d es SAGRADO y COMÚN a los 3** — `VAL_DAYS`, `V_IN_SHARED`,
   `RANDOM_SEED`, `DATA_DIR` se acuerdan UNA VEZ en `utils.py` y nadie los toca en su
   notebook. Cambiarlos invalida la comparación de backtests entre miembros.
5. **RMSE penaliza A y F** — invertir el esfuerzo ahí; en B/E un baseline puede bastar.
6. **Medir en espacio de PRECIOS** — entrenar en log-rets, pero reportar RMSE sobre precios
   reconstruidos. Baselines y NN se comparan en la misma escala (precios).
7. **El rollout no hace trampa** — solo el último precio conocido + predicciones propias.
   Verificado en `predict_autoregressive` (sin leakage de valores reales del futuro).
8. **Un enfoque por índice** — no forzar un único modelo global; el mejor de cada serie.
9. **Comparar con baselines** — flat/drift/random_walk antes de celebrar cualquier NN.
10. **LOSS configurable, RMSE como métrica** — entrenar con `'mse'` (default) o `'mae'`;
    NUNCA usar RMSE como loss (gradientes inestables). Decide el RMSE del backtest.

---

## Estructura del repo

```
TAREA_COMPETICION_REDES_NEURONALES/
├── CLAUDE.md                 ← este fichero
├── utils.py                  ← fontanería + CONSTANTES COMPARTIDAS
├── exp_TEMPLATE.ipynb        ← plantilla (no editar; copiar a exp_TUNOMBRE.ipynb)
├── exp_oscar.ipynb           ← notebook personal Oscar
├── exp_dani.ipynb            ← notebook personal Dani
├── exp_fernando.ipynb        ← notebook personal Fernando
├── COMPETICION.ipynb         ← entregable: consolidación por índice + CSV final
├── models/                   ← modelos exportados (<owner>_<Index>.keras)
├── results/                  ← JSONs por miembro (config + RMSE de backtest por índice)
├── data/                     ← soltar los CSV aquí el sábado
├── docs/                     ← documentación de contexto (ver tabla abajo)
└── notebooks_tarea/          ← notebooks de la tarea previa (referencia histórica)
```

---

## API de utils.py

### Constantes compartidas (acordar al inicio, NADIE las cambia luego)
```python
DATA_DIR    = 'data/'
N_DAYS_PRED = 252                 # horizonte de producción
VAL_DAYS    = 252                 # validación interna (= horizonte)
INDEX_COLS  = ['Index_A', ..., 'Index_F']
RANDOM_SEED = 42
V_IN_SHARED = 20                  # ⚠️ PROVISIONAL — reacordar al ver los datos
                                  #    (si las series son cortas, 20 es excesivo)
```

### Funciones por grupo
| Grupo | Funciones |
|-------|-----------|
| Carga | `load_hackathon_data(data_dir)` → dict de DataFrames (robusto a ficheros ausentes) |
| Preprocesado | `precios_a_logret`, `logret_a_precios`, `make_window_dataset(use_log_rets=True)`, `make_temporal_split`, `apply_ffd`, `align_aux_features` |
| Métricas | `rmse_per_index`, `rmse_mean` (oficial), `eval_directional` (secundaria) |
| **Validación** | **`backtest_autoregressive`** (el juez), `eval_all_baselines` |
| Rollout | `predict_autoregressive` (con clip defensivo anti-divergencia) |
| Baselines | `baseline_flat`, `baseline_drift`, `baseline_random_walk` |
| Detective | `lagged_correlation` (para el Ghost) |
| Modelos NN | `build_model` (lstm default), `compile_model(loss=...)`, `train_model`, `train_ensemble` |
| Entrega | `generar_submission`, `validar_submission` |
| Visualización | `plot_history`, `plot_rollout`, `plot_rmse_by_index` |

### Patrón de uso (por índice)
```python
serie = idx['Index_A'].dropna().values
X, y = make_window_dataset(serie[:-VAL_DAYS], V_IN, use_log_rets=True)
model = build_model('lstm', V_IN, loss=LOSS)
train_model(model, X[:cut], y[:cut], X[cut:], y[cut:])
bt = backtest_autoregressive(lambda x: model.predict(x, verbose=0).ravel()[0],
                             serie, log_ret_mode=True)   # RMSE en precios
```

---

## Documentación de contexto

| Fichero | Qué contiene | Cuándo |
|---------|-------------|--------|
| `docs/problema_autoregresivo.md` | **Spec del nuevo problema**, playbook por índice, reconstrucción log-ret→precio, no-leakage | **Primero** |
| `docs/modelos_fundacionales.md` | Zero-shot Chronos-2/TimesFM, GPU workaround | Alternativa zero-shot al rollout NN |
| `docs/transformers_y_huggingface.md` | PatchTST/iTransformer, HuggingFace Hub | Experimentos avanzados |
| `docs/entrenamiento_y_buenas_practicas.md` | Loop custom NN, callbacks, ensemble de seeds | Durante el entrenamiento |
| `docs/preprocesado_y_datos.md` | Log-retornos, split temporal, FFD | Al configurar datos |
| `docs/fundamentos_teoria.md` | 4 componentes ML, optimizadores, lr | Justificar diseño |
| `docs/resumen_tarea.md` | Tarea previa (MAE, no autorregresiva) — referencia histórica | Contexto de baseline NN |
