# HACKATHON — Predicción de Precios / Series Temporales

## Contexto
- **Tarea**: predecir una serie de precios a varios días en el futuro con NN + modelos fundacionales
- **Métrica de competición**: MAE (+ accuracy direccional como señal secundaria)
- **Stack**: Keras / TensorFlow · Chronos-2 · TimesFM · pandas · numpy · Python 3.11/3.12
- **Tiempo disponible**: 4 horas

## ⚠️ WORKAROUND GPU — EJECUTAR SIEMPRE AL INICIO

RTX 5070 Ti (Blackwell) es **incompatible** con TensorFlow GPU.
Esta línea va **ANTES de cualquier import de TF/Keras** o el proceso se cuelga:

```python
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"   # CPU-only
import tensorflow as tf   # solo después
```

Entorno de referencia (Daniel): Python 3.11.9 en `C:\venv_redes`.

---

## Estrategia en dos capas

El hackathon cubre lo explicado en clase: modelos fundacionales (zero-shot → fine-tuning)
**y** custom NN entrenadas desde cero. Trabajar en este orden:

```
CAPA 1 — Modelos fundacionales (objetivo: baseline sólido en < 30 min)
  1. Zero-shot con Chronos-2 (multi-activo) o TimesFM (univariado)
  2. Comparar entre ellos y con naive / media móvil
  3. Si hay tiempo: fine-tuning del mejor

CAPA 2 — Custom NN desde cero (complementario / si el enunciado lo pide)
  4. Barrido de 4 modelos (dense/lstm/cnn1d/cnn_lstm) con utils.py
  5. Ensemble de semillas del ganador
  6. FFD(d=0.2) si V_out = 1 día
```

---

## Flujo de trabajo rápido

```
1. Soltar CSV en data/
2. WORKAROUND GPU (línea os.environ antes de cualquier import)
3. Zero-shot Chronos-2 → baseline en minutos
4. Comparar con TimesFM y con naive
5. Abrir entrenamiento.ipynb → barrido custom NN
6. Ensemble del mejor modelo custom NN
7. Comparar todas las apuestas → escoger la mejor
```

---

## Estructura del repo

```
TAREA_COMPETICION_REDES_NEURONALES/
├── CLAUDE.md                        ← este fichero
├── utils.py                         ← custom NN: modelos, entrenamiento, evaluación
├── entrenamiento.ipynb              ← notebook custom NN (Capa 2)
├── data/                            ← soltar el CSV aquí al llegar
├── docs/                            ← documentación de contexto (ver abajo)
└── notebooks_tarea/tarea_previa/    ← notebooks de la tarea previa (referencia)
```

---

## Documentación de contexto

| Fichero | Qué contiene | Cuándo consultar |
|---------|-------------|-----------------|
| `docs/modelos_fundacionales.md` | Zero-shot con Chronos-2/TimesFM, catálogo de modelos, transfer learning, functional API, GPU workaround | **Primero** — estrategia principal del hackathon |
| `docs/entrenamiento_y_buenas_practicas.md` | Loop custom NN, callbacks, diagnóstico curvas (6 patrones), ensemble de seeds, lo que no funciona | Durante el entrenamiento custom |
| `docs/preprocesado_y_datos.md` | Log-retornos, split temporal, ventanas deslizantes, FFD(d=0.2) por horizonte, qué normalización destruye | Al configurar los datos |
| `docs/fundamentos_teoria.md` | Los 4 componentes ML, funciones de coste, optimizadores, lr empírico vs teórico, Functional API | Para justificar decisiones de diseño |
| `docs/resumen_tarea.md` | 256 experimentos de la tarea previa, resultados numéricos completos, errores a no repetir | Referencia de MAE baseline para custom NN |

---

## Palancas para reducir MAE — por orden de prioridad

### 1. Zero-shot con modelo fundacional — PRIMER EXPERIMENTO
Sin entrenamiento propio, gratis y rápido. Ver `docs/modelos_fundacionales.md`.

```python
# Multi-activo → Chronos-2
from chronos import BaseChronosPipeline
import torch
pipeline = BaseChronosPipeline.from_pretrained("amazon/chronos-t5-base", device_map="cpu")
forecast = pipeline.predict(context, prediction_length=N_DIAS, num_samples=20)
mediana = forecast.median(dim=1).values
```

### 2. Ensemble de semillas (custom NN) — PALANCA PRINCIPAL CAPA 2
Las diferencias entre semillas son del orden del ruido de inicialización.
Promediar 5 seeds elimina ese ruido sistemáticamente:

```python
result = train_ensemble(
    'lstm', X_tr, y_tr, X_v, y_v, X_ts, y_ts,
    V_in=V_IN, n_features=N_FEAT, n_targets=N_TARGETS,
    n_seeds=5, epochs=300
)
```

### 3. FFD(d=0.2) como preprocesado — solo si V_OUT = 1
La única mejora real medida en la tarea previa: −8.9% MAE.

```python
X_src, y_src = load_data('data/precios.csv', ffd_d=0.2)  # SOLO si V_OUT=1
```

**ADVERTENCIA**: para V_out ≥ 5 empeora hasta +45%. Ver `docs/preprocesado_y_datos.md`.

### 4. Elección de modelo custom NN
Probar los 4 siempre; quedarse con el mejor antes del ensemble:

| Modelo | lr | Notas |
|--------|-----|-------|
| `dense` | 1e-4 | baseline rápido |
| `lstm` | 3e-4 | mejor ratio params/MAE histórico |
| `cnn1d` | 3e-4 | captura patrones locales |
| `cnn_lstm` | 3e-4 | híbrido |

### 5. Modelo funcional (si hay datos de distinta naturaleza)
Si el enunciado da series + indicadores escalares → arquitectura multi-rama.
Ver sección 6 de `docs/modelos_fundacionales.md`.

### 6. Fine-tuning del modelo fundacional (si sobra tiempo)
Congelar primeras capas, reentrenar solo el final. Ver niveles 1-3 en
`docs/modelos_fundacionales.md`.

---

## Reglas de oro — NO negociables

1. **GPU workaround SIEMPRE primero** — `os.environ["CUDA_VISIBLE_DEVICES"] = "-1"` antes de cualquier import de TF
2. **Split temporal sin shuffle** — `make_splits` lo garantiza; nunca `shuffle=True`
3. **MAE como loss** — `compile_model` lo fija; no cambiar a MSE
4. **Reportar MAE + accuracy direccional** — `evaluate()` devuelve ambas
5. **No normalizar sin testear** — StandardScaler +4.1%, RollingZ +2.4% (medido)
6. **FFD solo para V_out=1** — para horizontes largos usa retornos crudos
7. **Mirar cuantiles de los fundacionales** — intervalo ancho = predicción poco fiable
8. **Comparar con baselines clásicos** — naive, media móvil antes de celebrar

---

## Resultados de referencia (tarea previa, 23 activos SP500)

| V_out | Naive | Lineal | Custom NN | FFD d=0.2 |
|-------|-------|--------|-----------|-----------|
| 1d | 0.0178 | 0.0124 | 0.0123 | **0.0112** |
| 5d | 0.0137 | 0.0056 | 0.0056 | — |
| 30d | 0.0125 | 0.0023 | 0.0023 | — |
| 90d | 0.0122 | 0.0013 | 0.0013 | — |

Diferencias entre arquitecturas NN: Δtest < 0.0001. Real, consistente, y es donde se compite.

---

## API de utils.py (custom NN)

| Función | Firma resumida | Uso |
|---------|---------------|-----|
| `load_data` | `(filepath, price_cols, return_cols, ffd_d)` | Carga CSV → (X_src, y_src) |
| `apply_ffd` | `(log_prices_df, d=0.2)` | FFD sobre log-precios |
| `create_time_series_data` | `(data, V_in, V_out)` | Ventanas deslizantes → X, y |
| `make_splits` | `(X, y)` | Split 72/18/10% cronológico sin shuffle |
| `build_model` | `(tipo, V_in, n_features, n_targets, units, dropout, lr)` | dense/lstm/cnn1d/cnn_lstm |
| `train_model` | `(model, X_tr, y_tr, X_v, y_v, epochs, batch_size)` | Fit + restore best weights |
| `train_ensemble` | `(tipo, ..., n_seeds=5)` | Ensemble de semillas |
| `evaluate` | `(model, X_tr, X_v, X_ts, ...)` | Dict MAE/RMSE/dir en 3 splits |
| `eval_mae_naive` | `(X, y)` | Baseline último valor conocido |

## Si el dataset es diferente al de la tarea previa

- **Precios directos** (no retornos): `load_data(..., price_cols=['Close', ...])`
- **Ya son retornos**: `load_data(..., return_cols=['ret_SPY', ...])`
- **Target univariante** (una sola columna): `build_model(..., n_targets=1)`
- **Ventana desconocida**: empezar con V_IN=10 y probar [5, 30]
- **Varios activos + indicadores escalares**: modelo funcional multi-rama (ver docs)
