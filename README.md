# Hackathon — Predicción de Retornos de Activos

Base para el hackathon de redes neuronales (regresión de retornos, Keras/TF).  
**Equipo**: Oscar · Dani · Fernando

## Arranque rápido

```bash
pip install -r requirements.txt
```

## Flujo del equipo (sábado)

```
ANTES DE EMPEZAR (5 min, juntos):
  1. Soltar el CSV en data/
  2. Leer el enunciado y rellenar el checklist del notebook
  3. Acordar V_IN, V_OUT, FFD_D → editar en utils.py (una sola vez)

TRABAJO PARALELO (cada uno):
  4. Abrir exp_oscar.ipynb / exp_dani.ipynb / exp_fernando.ipynb
  5. Ejecutar secciones 0-7 completas
  6. Sección 7 exporta models/<nombre>.keras + results/<nombre>.json

SÍNTESIS FINAL:
  7. Abrir COMPETICION.ipynb → carga los 3 modelos → ensemble automático
```

## Ficheros clave

| Fichero | Descripción |
|---------|-------------|
| `utils.py` | Funciones compartidas + **constantes de datos** (`FILEPATH_SHARED`, `V_IN_SHARED`, `V_OUT_SHARED`…) |
| `exp_oscar.ipynb` | Notebook personal Oscar: cribado → entreno → ensemble → exportar modelo |
| `exp_dani.ipynb` | Notebook personal Dani |
| `exp_fernando.ipynb` | Notebook personal Fernando |
| `COMPETICION.ipynb` | Entregable final: carga los 3 modelos, verifica compatibilidad, ensemble automático |
| `models/` | Modelos exportados (`oscar.keras`, `dani.keras`, `fernando.keras`) |
| `results/` | JSONs de config y MAE de cada miembro |
| `CLAUDE.md` | Contexto completo: palancas de mejora, reglas de oro, API de utils.py |
| `docs/resumen_tarea.md` | Referencia de la tarea previa (arquitecturas, resultados numéricos) |

## Regla crítica

Las constantes `FILEPATH_SHARED`, `V_IN_SHARED`, `V_OUT_SHARED`, `FFD_D_SHARED` de `utils.py`
se acuerdan entre los 3 al inicio y **nadie las cambia en su notebook personal**.
Cambiarlas invalida el ensemble porque los splits dejan de ser los mismos.

## Familias de modelos disponibles

| Familia | Cómo | Dónde |
|---------|------|-------|
| Dense / LSTM / CNN / CNN+LSTM | `build_model(tipo, ...)` | `utils.py` + secciones 3-5 de cada notebook |
| Fundacionales zero-shot | Chronos-2, TimesFM | `docs/modelos_fundacionales.md` |
| Transformers especializados | PatchTST, iTransformer | `docs/transformers_y_huggingface.md` |
| HuggingFace Hub (búsqueda) | `list_models(task='time-series-forecasting')` | `docs/transformers_y_huggingface.md` |
| Funcional multi-rama | Keras Functional API | `docs/modelos_fundacionales.md` §6 |

## Palancas principales

1. **Ensemble de los 3 modelos** (`COMPETICION.ipynb`) — diversidad entre miembros reduce varianza
2. **Ensemble de semillas** (`train_ensemble`, `n_seeds=5`) — reduce ruido de inicialización
3. **FFD(d=0.2)** en `load_data(ffd_d=0.2)` — −8.9 % MAE, **solo para V_out=1**
4. Barrido de los 4 modelos: `dense`, `lstm`, `cnn1d`, `cnn_lstm`
5. Buscar en HuggingFace si sobra tiempo: `huggingface.co/models?pipeline_tag=time-series-forecasting`

Ver `CLAUDE.md` para instrucciones detalladas.

## Stack

Python 3.11/3.12 · Keras 3.x (backend TensorFlow) · CPU-only (workaround RTX 5070 Ti)
