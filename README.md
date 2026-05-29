# Hackathon — Predicción de Retornos de Activos

Base mínima para el hackathon de redes neuronales (regresión de retornos, Keras/TF).

## Arranque rápido

```bash
pip install -r requirements.txt
```

1. Soltar el CSV del hackathon en `data/`
2. Abrir `entrenamiento.ipynb`
3. Ajustar la celda marcada con `# ── AJUSTAR AQUÍ ──`
4. Run All

## Ficheros clave

| Fichero | Descripción |
|---------|-------------|
| `entrenamiento.ipynb` | Notebook de trabajo: carga → split → barrido → ensemble |
| `utils.py` | Funciones compartidas: modelos, entrenamiento, evaluación, FFD |
| `CLAUDE.md` | Contexto del hackathon, palancas de mejora, reglas de oro |
| `docs/resumen_tarea.md` | Referencia de la tarea previa (arquitecturas, resultados) |
| `notebooks_tarea/tarea_previa/` | Notebooks originales (00–08) como referencia |

## Palancas principales

1. **Ensemble de semillas** (`train_ensemble`, `n_seeds=5`) — reduce varianza
2. **FFD(d=0.2)** en `load_data(ffd_d=0.2)` — −8.9 % MAE para V_out=1
3. Barrido de los 4 modelos: `dense`, `lstm`, `cnn1d`, `cnn_lstm`

Ver `CLAUDE.md` para instrucciones detalladas.

## Stack

Python 3.12 · Keras 3.x (backend TensorFlow)
