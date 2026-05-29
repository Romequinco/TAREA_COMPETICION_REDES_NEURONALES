# Hackathon — Predicción autorregresiva de 6 índices financieros

Predicción día a día de 6 índices (252 pasos autorregresivos), métrica RMSE promedio.  
**Equipo**: Oscar · Dani · Fernando

## Arranque rápido

```bash
pip install -r requirements.txt
```

## Flujo del equipo (sábado)

```
ANTES DE EMPEZAR (5 min, todos):
  1. Soltar los CSV en data/
  2. Ejecutar 00_carga_y_EDA.ipynb — diagnóstico y detective (Ghost lag, macro-C, network-F)
  3. Acordar V_IN_SHARED viendo las series reales → editar UNA VEZ en utils.py

TRABAJO PARALELO:
  4. 01_baselines.ipynb — baselines para los 6 índices (asegurar el aprobado)
  5. Notebooks de índice (03-08) — OWNER = "nombre" al inicio de cada uno
     Cada uno guarda results/index_X.json (única fuente de verdad por índice)

SÍNTESIS FINAL:
  6. 09_consolidacion.ipynb — lee los 6 JSON, genera 252×6, valida, CSV de entrega
     Fallback: df_pred.to_clipboard() → pegar en Excel del profesor
```

## Ficheros clave

| Fichero | Descripción |
|---------|-------------|
| `utils.py` | Funciones compartidas + **constantes** (`V_IN_SHARED`, `VAL_DAYS`, `RANDOM_SEED`…) |
| `00_carga_y_EDA.ipynb` | Diagnóstico + DETECTIVE: Ghost lag, correlación macro-C, network-F |
| `01_baselines.ipynb` | Flat/drift/random_walk → `results/baselines.json` |
| `02_sentiment_news.ipynb` | ⚠️ Baja prioridad — sentiment con FinBERT/multilingüe |
| `03_index_A.ipynb` | Alpha-Tech — ALTO esfuerzo, LSTM + ensemble |
| `04_index_B.ipynb` | Steady-State — BAJO esfuerzo, baseline plano |
| `05_index_C.ipynb` | Energy-Pulse — MEDIO, LSTM + macro_factors |
| `06_index_D.ipynb` | The Ghost — BAJO-MEDIO, detective + replicar índice fuente con lag |
| `07_index_E.ipynb` | Global-ESG — MEDIO, LSTM o baseline |
| `08_index_F.ipynb` | Digital-Frontier — ALTO esfuerzo, LSTM + network_metrics |
| `09_consolidacion.ipynb` | Lee 6 JSON → df 252×6 → validar → CSV final + fallback manual |
| `models/` | Modelos exportados (`{owner}_{Index}.keras`) |
| `results/` | `baselines.json` + `index_A…F.json` (schema acordado) |

## Regla crítica

Las constantes de `utils.py` se acuerdan **una vez al inicio** y nadie las cambia en su notebook.
Cada `results/index_X.json` lo escribe **exclusivamente** el notebook de ese índice.
`09_consolidacion.ipynb` solo lee — nunca escribe JSON de índice.

## Stack

Python 3.11 · Keras/TensorFlow (CPU-only, workaround RTX 5070 Ti) · pandas · numpy

## Documentación de contexto (`docs/`)

| Fichero | Cuándo consultar |
|---------|-----------------|
| `problema_autoregresivo.md` | **Primero** — spec del problema, playbook por índice, detalles del enunciado oficial |
| `modelos_fundacionales.md` | Zero-shot Chronos-2/TimesFM, GPU workaround |
| `transformers_y_huggingface.md` | PatchTST, iTransformer, HuggingFace Hub |
| `entrenamiento_y_buenas_practicas.md` | Callbacks, ensemble de seeds, diagnóstico de curvas |
| `preprocesado_y_datos.md` | Log-retornos, split temporal, FFD |
| `fundamentos_teoria.md` | Funciones de coste, optimizadores, lr |
| `resumen_tarea.md` | Tarea previa (MAE, no autorregresiva) — referencia histórica |
