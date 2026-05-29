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
  2. Leer el enunciado → rellenar el checklist del notebook
  3. Acordar V_IN, V_OUT, FFD_D → editar en utils.py (una sola vez, todos usan los mismos)

TRABAJO PARALELO — cada uno en su exp_*.ipynb:
  § 0   Diagnóstico automático del dataset
  § 1   Configuración (constantes de datos de utils.py + parámetros personales)
  § 2   Ventanas deslizantes + split temporal
  § 3   Cribado rápido: 4 arquitecturas custom (dense / lstm / cnn1d / cnn_lstm)
  § 4   Entrenamiento completo de los mejores candidatos
  § 5   Ensemble de semillas del ganador
  § 6   Tabla de resultados → compartir números con el equipo
  § 8   Experimentos adicionales (opcional, si queda tiempo):
          - Transformers: PatchTST, iTransformer  (docs/transformers_y_huggingface.md)
          - Fundacionales: Chronos-2, TimesFM, Kronos  (docs/modelos_fundacionales.md)
          - Funcional multi-rama si hay covariables escalares
          - Buscar en HuggingFace Hub modelos de forecasting financiero
        → Rellenar tabla A/B/C y decidir qué modelo exportar
  § 7   Exportar 1 modelo: models/<nombre>.keras + results/<nombre>.json
          (por defecto: ganador de §4; si §8 dio algo mejor → ejecutar
           primero "Decisión final" de §8 para cambiar mejor_modelo)

SÍNTESIS FINAL:
  Abrir COMPETICION.ipynb → verifica compatibilidad → ensemble automático → MAE entregable
  (Si algo falla: sección de fallback manual ya documentada)
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
| `docs/` | Documentación técnica (ver tabla abajo) |
| `notebooks_tarea/tarea_previa/` | Notebooks originales 00–08 de la tarea anterior (referencia) |

## Documentación de contexto (`docs/`)

| Fichero | Cuándo consultar |
|---------|-----------------|
| `modelos_fundacionales.md` | **Primero** — zero-shot Chronos-2/TimesFM, transfer learning, modelo funcional, GPU workaround |
| `transformers_y_huggingface.md` | PatchTST, iTransformer, cómo buscar y cargar modelos de HuggingFace Hub |
| `entrenamiento_y_buenas_practicas.md` | Callbacks, diagnóstico de curvas, ensemble de seeds, lo que no funciona |
| `preprocesado_y_datos.md` | Log-retornos, split temporal, FFD(d=0.2) por horizonte, qué normalización destruye |
| `fundamentos_teoria.md` | Funciones de coste, optimizadores, lr empírico vs teórico, Functional API |
| `resumen_tarea.md` | 256 experimentos de la tarea previa, MAE de referencia, errores a no repetir |

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
