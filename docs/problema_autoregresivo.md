# Problema autorregresivo — Spec y playbook del hackathon

Documento de cabecera del hackathon. Leer **antes** que los demás docs (que son
referencia técnica de la tarea previa, todavía válida pero no específica de este problema).

---

## 1. Qué cambió respecto a la tarea previa

| | Tarea previa | **Hackathon (este problema)** |
|---|---|---|
| Predicción | 1 paso (promedio de V_out días) | **252 pasos autorregresivos, día a día** |
| Realimentación | no | **sí — cada predicción es input del siguiente** |
| Métrica | MAE | **RMSE medio sobre 6 índices** |
| Target | 23 activos a la vez | **6 índices, un enfoque por índice** |
| Datos | solo precios | precios + news + macro + network |
| Error | independiente por muestra | **se acumula en el rollout** |

El cambio de MAE→RMSE no es cosmético: RMSE eleva el error al cuadrado, así que un
índice volátil con error 30 000 aporta 9·10⁸ al promedio, mientras uno defensivo con
error 6 aporta 36. **El ranking lo deciden A y F.** Optimizar B/C/D/E al límite no mueve
la aguja si A/F están mal.

---

## 2. Rollout autorregresivo: el núcleo

```
día 1: modelo(ventana_real)        -> pred_1,  ventana <- [..., pred_1]
día 2: modelo(ventana con pred_1)  -> pred_2,  ventana <- [..., pred_2]
...
día 252: modelo(ventana con preds previas) -> pred_252
```

El modelo nunca vuelve a ver un dato real tras el día 0. Un error pequeño en el día 1
se propaga y amplifica. Por eso **un MAE/loss bajo en validación de 1 paso NO predice el
RMSE del rollout** — solo el backtest a 252d lo mide.

---

## 3. Log-retornos -> precios (por qué y cómo)

No conocemos la escala de los índices hasta el sábado (el umbral RMSE<75000 no la
revela). Una NN entrenada sobre precios crudos de escala desconocida (¿10²? ¿10⁶?) es
numéricamente inestable. Solución: **operar en log-retornos** (escala ~0.01, universal) y
**reconstruir precios** para medir.

```python
log_rets = precios_a_logret(precios)                 # diff(log(P)),  ~0.01
# ... el modelo predice log-rets ...
precios  = logret_a_precios(log_rets, precio_inicial) # P0 · exp(cumsum(log_rets))
```

- `precio_inicial` = **último precio real conocido** (antes del primer día a predecir).
- La reconstrucción es exacta (round-trip error ~1e-13).
- `make_window_dataset(..., use_log_rets=True)` entrena en este espacio.
- `backtest_autoregressive(..., log_ret_mode=True)` reconstruye y mide **RMSE en precios**.

### No-leakage (por qué el backtest es fiable)
En el rollout, la ventana se rellena **siempre** con la predicción propia del modelo.
El único dato real usado es `precio_inicial` (el ancla). Los precios reales del período
de validación se usan **solo al final** para calcular el RMSE, nunca como input. Si un
valor real se colara como input, el backtest mentiría — y con 6 entregas, un backtest que
miente es el peor fallo posible.

### Clip defensivo
`predict_autoregressive(clip_logret=0.5)` recorta cada log-ret predicho a ±0.5 antes de
acumularlo. Salvaguarda contra divergencia (`exp(cumsum)` desborda si el modelo se
desboca). Con clip + reconstrucción en float64, un modelo malo da un RMSE enorme pero
**finito** que el backtest reporta — en vez de `inf`/`nan` que rompería la comparación.
No sustituye al backtest: si el clip se activa, el modelo ya es malo.

---

## 4. Playbook por índice

| Índice | Pista del enunciado | Primer intento | Datos auxiliares |
|--------|--------------------|----------------|------------------|
| **A** | Alta volatilidad / crecimiento | LSTM log-ret + ensemble 5 seeds | — |
| **B** | Steady-State defensivo | `baseline_flat` (probablemente ya gana) | — |
| **C** | Energy-Pulse | LSTM + features macro | `macro_factors` (oro/crudo/tipos) |
| **D** | "The Ghost" | derivar de otro índice + lag | `lagged_correlation(rets, 'Index_D')` |
| **E** | Global-ESG | LSTM estándar o baseline | (news?) |
| **F** | Digital-Frontier | LSTM + features network + ensemble | `network_metrics` (on-chain) |

### Cazar el Ghost (D)
```python
rets = np.log(idx).diff().dropna()
corr = lagged_correlation(rets, 'Index_D', max_lag=30)
peak = corr.abs().stack().idxmax()       # (lag, columna) de correlación máxima
# Si peak da, p.ej., (5, 'Index_A'): D[t] ~ A[t-5]. Derivar D directamente puede
# batir a cualquier NN — y es casi gratis.
```

### Features auxiliares (C, F)
```python
aux = align_aux_features(idx, data['train_macro'], data['train_macro'].columns)
X, y = make_window_dataset(serie, V_IN, use_log_rets=True, aux_features=aux.values)
# En el rollout hay que pasar el aux del FUTURO (test_macro / test_network):
#   backtest_autoregressive(..., aux_data=aux_completo_alineado_con_la_serie)
```
⚠️ `aux_data` debe tener **la misma longitud** que la serie (el backtest lo comprueba con
un assert; un aux desalineado produce un RMSE falso silenciosamente).

---

## 5. Reglas de oro (resumen — versión completa en CLAUDE.md)

1. GPU workaround antes de importar TF.
2. **Backtest 252d antes de subir** — único juez fiable.
3. **6 entregas** — cada una una mejora medida, no una corazonada. `validar_submission` siempre.
4. **Split de 252d sagrado y común** — constantes en `utils.py`, nadie las cambia.
5. **Esfuerzo en A y F** — dominan el RMSE.
6. Medir en precios; entrenar en log-rets.
7. Rollout sin trampa (predicciones propias + ancla).
8. Un enfoque por índice.
9. Baselines antes de celebrar.
10. Loss `mse`/`mae` configurable; **nunca RMSE como loss**.

---

## 6. Reparto del equipo

- **Oscar** — infra + validación + entrega (`utils.py`, backtest, submission, `COMPETICION.ipynb`).
- **Miembro 2** — baselines + defensivos (B, E) + Ghost (D).
- **Miembro 3** — volátiles A y F (los que pesan) + features macro (C) y network (F) + ensembles.

Cada uno trabaja en su `exp_*.ipynb`; la consolidación recoge el mejor enfoque por índice
venga del notebook que venga.
