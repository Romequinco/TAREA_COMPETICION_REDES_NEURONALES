# HACKATHON — Predicción de Retornos de Activos

## Contexto
- **Tarea**: regresión multivariante de retornos financieros (SP500-style)
- **Métrica de competición**: MAE (+ accuracy direccional como señal secundaria)
- **Stack**: Keras / TensorFlow · pandas · numpy · scikit-learn · Python 3.12
- **Tiempo disponible**: 4 horas

## Mentalidad competitiva
En la tarea previa todos los modelos convergían al mismo MAE. **En competición esa
no es la actitud correcta.** Las diferencias entre arquitecturas son reales y
consistentes aunque pequeñas, y el ganador se decide en esos márgenes.
**El trabajo del sábado es arañar decimales y reducir varianza activamente.**
El margen competitivo vive en el epsilon — hay que minimizarlo.

## Flujo de trabajo rápido
```
1. Soltar CSV en data/
2. Abrir entrenamiento.ipynb
3. Celda "AJUSTAR AQUÍ": cambiar filepath y columnas
4. Run All  →  barrido de 4 modelos + ensemble en ~10-20 min
```

## Estructura del repo
```
TAREA_COMPETICION_REDES_NEURONALES/
├── CLAUDE.md                    ← este fichero
├── utils.py                     ← todo el código reutilizable
├── entrenamiento.ipynb          ← notebook de trabajo del sábado
├── data/                        ← soltar el CSV aquí al llegar
├── docs/                        ← documentación de contexto (ver abajo)
└── notebooks_tarea/tarea_previa/ ← notebooks originales (referencia)
```

## Documentación de contexto

Síntesis del conocimiento del máster relevante para el hackathon. Leer en orden
si necesitas refrescar fundamentos; usar como referencia rápida durante el trabajo.

| Fichero | Qué contiene | Cuándo consultar |
|---------|-------------|-----------------|
| `docs/resumen_tarea.md` | Arquitecturas probadas, resultados numéricos completos (256 experimentos), errores a no repetir | Antes de empezar; referencia de MAE baseline |
| `docs/fundamentos_teoria.md` | Los 4 componentes ML, funciones de coste por problema, comparativa de optimizadores, por qué Adam lr=3e-4 (y cuándo usar 1e-4), arquitecturas y cuándo elegir cada una | Si necesitas justificar una decisión de diseño |
| `docs/preprocesado_y_datos.md` | Log-retornos, datos NO-iid, split cronológico, ventanas deslizantes, FFD con tabla de resultados por horizonte, qué normalización destruye y por qué | Al configurar `load_data` y decidir si usar FFD |
| `docs/entrenamiento_y_buenas_practicas.md` | Loop de entrenamiento, callbacks (por qué no EarlyStopping), diagnóstico por curvas de aprendizaje (tabla de 6 patrones), regularización, ensemble de seeds paso a paso, lo que no funciona | Durante el entrenamiento; para diagnosticar problemas |

---

## Palancas para reducir MAE — por orden de impacto esperado

### 1. Ensemble de semillas — PALANCA PRINCIPAL
Las diferencias entre semillas del mismo modelo son del orden del ruido de
inicialización. Promediar N semillas elimina ese ruido sistemáticamente y es
la primera cosa a activar en competición:

```python
result = train_ensemble(
    'lstm', X_tr, y_tr, X_v, y_v, X_ts, y_ts,
    V_in=V_IN, n_features=N_FEAT, n_targets=N_TARGETS,
    n_seeds=5, epochs=300
)
print(result['mae_test'])
```

Intentar con n_seeds=3 primero si el tiempo aprieta; subir a 10 si sobra.

### 2. FFD(d=0.2) como preprocesado de entrada
La **única mejora real medida** en la tarea previa: −8.9 % MAE vs retornos crudos.
Aplica diferenciación fraccional sobre log-precios:

```python
X_src, y_src = load_data('data/precios.csv', ffd_d=0.2)
# X_src = serie FFD (input del modelo)
# y_src = retornos crudos alineados (target y)
```

**ADVERTENCIA**: solo mejora para V_out=1 día. Para V_out ≥ 5 empeora hasta +45 %.
Si el target es horizonte largo, usar retornos crudos (ffd_d=None).

### 3. Elección de modelo
Hay diferencias reales en los decimales aunque la zona de convergencia es estrecha.
Probar los 4 siempre y quedarse con el mejor antes del ensemble:

| modelo    | lr recomendado | notas                              |
|-----------|---------------|-------------------------------------|
| `dense`   | 1e-4          | rápido; baseline sólido             |
| `lstm`    | 3e-4          | mejor ratio params/MAE histórico    |
| `cnn1d`   | 3e-4          | captura patrones locales            |
| `cnn_lstm`| 3e-4          | híbrido; regularización implícita   |

### 4. Ajuste de hiperparámetros (si sobra tiempo)
- `units`: probar 32 / 64 / 128
- `epochs`: subir de 300 a 500 en el modelo ganador
- `dropout`: 0.1–0.2 en lstm y cnn_lstm para reducir varianza
- `V_IN`: probar [5, 10, 30] — en el problema original V_in no tenía impacto,
  pero puede variar con datos distintos

---

## Reglas de oro — NO negociables

1. **Split temporal sin shuffle siempre** — `make_splits` ya lo garantiza;
   no usar `shuffle=True` ni `random_state` que implique reordenación
2. **MAE como loss** — `compile_model` ya lo fija; no cambiar a MSE
3. **Reportar MAE + accuracy direccional** — `evaluate()` devuelve ambas
4. **No normalizar sin testear** — StandardScaler y Rolling Z-score empeoran
   en este tipo de problema (medido: +4.1 % y +2.4 % respectivamente)
5. **FFD solo para V_out=1** — para horizontes largos usar retornos crudos

---

## Resultados de referencia (tarea previa, 23 activos SP500)

| V_out | Naive   | Lineal  | NN base | FFD d=0.2 |
|-------|---------|---------|---------|-----------|
| 1d    | 0.0178  | 0.0124  | 0.0123  | **0.0112**|
| 5d    | 0.0137  | 0.0056  | 0.0056  | —         |
| 30d   | 0.0125  | 0.0023  | 0.0023  | —         |
| 90d   | 0.0122  | 0.0013  | 0.0013  | —         |

Las diferencias entre arquitecturas NN: Δtest < 0.0001 en 15/16 combinaciones.
Pero ese Δ es real, consistente y es donde se compite.

---

## API de utils.py

| Función | Firma resumida | Uso |
|---------|---------------|-----|
| `load_data` | `(filepath, price_cols, return_cols, ffd_d)` | Carga CSV → (X_src, y_src) |
| `apply_ffd` | `(log_prices_df, d=0.2)` | FFD sobre log-precios |
| `create_time_series_data` | `(data, V_in, V_out)` | Ventanas deslizantes → X, y |
| `make_splits` | `(X, y)` | Split 72/18/10 % cronológico |
| `build_model` | `(tipo, V_in, n_features, n_targets, units, dropout, lr)` | dense/lstm/cnn1d/cnn_lstm |
| `train_model` | `(model, X_tr, y_tr, X_v, y_v, epochs, batch_size)` | Fit + restore best weights |
| `train_ensemble` | `(tipo, ..., n_seeds=5)` | Ensemble de semillas → palanca principal |
| `evaluate` | `(model, X_tr, X_v, X_ts, y_tr, y_v, y_ts)` | Dict MAE/RMSE/dir en 3 splits |
| `eval_mae_naive` | `(X, y)` | Baseline último valor conocido |
| `build_results_df` | `(results_dict)` | Dict → DataFrame MultiIndex |
| `plot_history` | `(hist, title)` | Curva loss/val_loss |
| `plot_mae_matrix` | `(mat_df, title)` | Heatmap 4×4 |

## Si el dataset es diferente al de la tarea previa

- **Target es una sola columna** (regresión univariante):
  `build_model(..., n_targets=1)` y `train_ensemble(..., n_targets=1)`
- **CSV ya tiene retornos** (no precios):
  `load_data(..., return_cols=['ret_SPY', 'ret_AAPL', ...])`
- **Solo precios de cierre** en columna 'Close':
  `load_data(..., price_cols=['Close'])`
- **Ventana de entrada desconocida**: empezar con V_IN=10 y probar [5, 30]
