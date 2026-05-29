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
  2. Ejecutar 00_carga_y_EDA.ipynb — diagnóstico + detective (Ghost, macro-C, network-F)
  3. Acordar V_IN_SHARED viendo las series reales → editar UNA VEZ en utils.py

TRABAJO PARALELO (cada miembro en sus notebooks asignados):
  4. 01_baselines.ipynb — baselines para los 6 índices → results/baselines.json
  5. Notebooks de índice (03-08) — un notebook por índice, OWNER = "nombre" al inicio
     Cada uno guarda results/index_X.json (UNA SOLA FUENTE DE VERDAD)
  6. Backtest 252d SIEMPRE antes de guardar el JSON

SÍNTESIS FINAL:
  7. 09_consolidacion.ipynb — lee los 6 JSON, genera df 252×6, validar, CSV
     Fallback manual: df_pred.to_clipboard() → pegar en Excel del profesor
```

### Reparto sugerido
- **Oscar** — infra + validación + entrega: `utils.py`, `00_carga_y_EDA`, `01_baselines`, `09_consolidacion`.
- **Miembro 2** — índices defensivos + Ghost: `04_index_B`, `07_index_E`, `06_index_D`.
- **Miembro 3** — volátiles + datos auxiliares: `03_index_A`, `08_index_F`, `05_index_C`.

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
├── CLAUDE.md                   ← este fichero
├── utils.py                    ← fontanería + CONSTANTES COMPARTIDAS (NO tocar)
├── 00_carga_y_EDA.ipynb        ← diagnóstico + DETECTIVE (Ghost, macro-C, network-F)
├── 01_baselines.ipynb          ← flat/drift/rw → results/baselines.json
├── 02_sentiment_news.ipynb     ← ⚠️ BAJA PRIORIDAD — sentiment con transformer
├── 03_index_A.ipynb            ← Alpha-Tech (ALTO esfuerzo, LSTM+ensemble)
├── 04_index_B.ipynb            ← Steady-State (BAJO esfuerzo, baseline plano)
├── 05_index_C.ipynb            ← Energy-Pulse (MEDIO, LSTM+macro)
├── 06_index_D.ipynb            ← The Ghost (BAJO-MEDIO, detective+lag)
├── 07_index_E.ipynb            ← Global-ESG (MEDIO, LSTM o baseline)
├── 08_index_F.ipynb            ← Digital-Frontier (ALTO esfuerzo, LSTM+network)
├── 09_consolidacion.ipynb      ← lee 6 JSON → df 252×6 → CSV → validar
├── models/                     ← modelos exportados ({owner}_{Index}.keras)
├── results/                    ← baselines.json + index_A…F.json
├── data/                       ← soltar los CSV aquí el sábado
└── docs/                       ← documentación de contexto (ver tabla abajo)
```

### Contrato entre notebooks — una fuente de verdad por JSON
- `01_baselines.ipynb` escribe SOLO `results/baselines.json`.
- Cada `results/index_X.json` lo escribe EXCLUSIVAMENTE `0X_index_X.ipynb`, incluso si el approach ganador es un baseline.
- `09_consolidacion.ipynb` **solo lee** los JSON — nunca los escribe.
- **Ancla de reconstrucción:** `precio_inicial` = `train_indices[col].iloc[-1]` para todos los índices, en todos los notebooks.
- **CLIP_LOGRET = 0.5** — `predict_autoregressive` recorta cada log-ret a ±0.5 antes de acumularlo. Salvaguarda anti-divergencia activa por defecto. Si el clip se activa el modelo ya es malo.

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
