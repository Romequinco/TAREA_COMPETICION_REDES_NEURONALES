# Preprocesado y Datos — Síntesis MIAX B3

Síntesis de: `preprocesado-datos.md`, `b3_s2_mapa.md`, `b3_s1_mapa.md` (sección datos).
Énfasis en hallazgos empíricos de la tarea previa: qué funciona, qué rompe y por qué.

---

## 1. Por qué log-retornos (no precios)

Los precios de activos siguen aproximadamente una caminata aleatoria (no-estacionarios).
Los log-retornos son estacionarios y tienen propiedades estadísticas manejables:

```
rₜ = log(Pₜ) - log(Pₜ₋₁)  ≈  (Pₜ - Pₜ₋₁) / Pₜ₋₁
```

**Ventajas**:
- Estacionarios: media ≈ 0, varianza aproximadamente constante en el tiempo
- Aditivos: r(a→c) = r(a→b) + r(b→c), lo que simplifica el análisis multi-período
- Simétricos: log(1.1) ≈ −log(1/1.1), a diferencia de retornos aritméticos
- Misma escala para todos los activos independientemente de su precio absoluto

**En código**:
```python
returns = np.log(precios).diff().dropna()
```

---

## 2. Datos financieros son NO-i.i.d. — la regla más importante

Los supuestos estándar de ML (datos IID) no se cumplen en series temporales financieras:

- **Autocorrelación temporal**: el retorno de hoy tiene correlación (débil pero real) con
  retornos pasados
- **Heteroscedasticidad**: la varianza cambia en el tiempo (clusters de volatilidad: las
  crisis se agrupan)
- **Overlapping outcomes**: si V_out=30, la muestra del día t y la del día t+1 comparten
  29 de los 30 días del target → no son independientes

**Consecuencia**: si mezclas el orden cronológico (`shuffle=True`), datos de fechas
futuras contaminan el entrenamiento. El modelo "aprende" el futuro de sus propios datos.
El MAE test será artificialmente bajo y completamente inútil como predictor real.

**Regla absoluta**: `shuffle=False` siempre. Sin excepciones.

---

## 3. La partición cronológica: 72 / 18 / 10

```
|─────────────── train (~72%) ───────────────|──── val (~18%) ────|─ test (10%) ─|
  T₀ ─────────────────────────────────────── T₁ ──────────────── T₂ ─────────── T₃
                                              ↑ sin shuffle en ningún paso
```

**Por qué estos porcentajes**:
- **10% test**: estándar del enunciado; con ~16K días y V_in=10 da ~1600 muestras de test
- **25% de train_full → val** (18% del total): el 5% original del profesor era insuficiente
  para dar señal robusta a `ReduceLROnPlateau` (solo ~800 muestras). Con 25%, tenemos
  ~2900 muestras de validación, suficiente para que el plateau sea significativo

**En código** (`make_splits` en utils.py):
```python
# Paso 1: 90% train_full / 10% test
X_tr_full, X_ts, y_tr_full, y_ts = train_test_split(X, y, test_size=0.10, shuffle=False)
# Paso 2: 75% train / 25% val (del train_full)
X_tr, X_v, y_tr, y_v = train_test_split(X_tr_full, y_tr_full, test_size=0.25, shuffle=False)
# Resultado: ~72% train | ~18% val | 10% test
```

El `random_state=42` en `make_splits` solo afecta a la partición determinística con
`shuffle=False`; no introduce aleatoriedad real en el orden.

---

## 4. Ventanas deslizantes: cómo construye `create_time_series_data`

La función genera pares (X, y) con una ventana deslizante de paso 1:

```
datos: [r₁, r₂, r₃, ..., rₙ]

Para cada posición i:
  X[i] = datos[i : i+V_in]               # shape: (V_in, n_features)
  y[i] = mean(datos[i+V_in : i+V_in+V_out])   # shape: (n_features,)
```

El output `y` es el **promedio de los V_out retornos futuros**, no el retorno en un solo
día. Para V_out=1 coincide con el retorno del día siguiente exactamente.

**Volumen de datos con 16K días**:

| V_in | V_out | N muestras | Train | Val | Test |
|------|-------|-----------|-------|-----|------|
| 10 | 1 | 15.989 | 11.630 | 2.908 | 1.451 |
| 10 | 30 | 15.960 | 11.621 | 2.905 | 1.434 |
| 90 | 90 | 15.820 | 11.515 | 2.879 | 1.426 |

Con ventanas grandes (V_in=90, V_out=90) se pierden ~180 días de los extremos, pero el
volumen sigue siendo suficiente.

---

## 5. FFD (Diferenciación Fraccional): el único preprocesado que funciona

### Fundamento teórico

La diferenciación clásica (d=1 → log-retornos) es estacionaria pero destruye toda la
memoria a largo plazo del precio. FFD (López de Prado, cap. 5) aplica d ∈ (0,1):

```
xₜ^(d) = Σₖ wₖ · xₜ₋ₖ,  con  wₖ = (-1)^k · Γ(d+1) / (k! · Γ(d-k+1))
```

Los pesos decaen en función de k (los datos recientes pesan más que los lejanos), pero
con d<1 nunca llegan a cero completamente, manteniendo memoria fraccional.

- d=1: retornos clásicos (máxima estacionariedad, mínima memoria)
- d=0: serie original (máxima memoria, no estacionaria)
- d=0.2: **balance óptimo** medido empíricamente en nuestra tarea

### Resultados empíricos: barrido de d (V_in=30, V_out=1)

| d | MAE test | Δ vs crudo (0.0123) | |
|---|---------|---------------------|-|
| 0.1 | 0.0122 | −0.8% | |
| **0.2** | **0.0112** | **−8.9%** | ← ÓPTIMO |
| 0.3 | 0.0113 | −8.1% | |
| 0.4 | 0.0114 | −7.3% | |
| 0.5 | 0.0116 | −5.7% | |
| 0.6 | 0.0124 | +0.8% | |
| 1.0 (= retornos) | 0.0124 | +0.8% | |

La curva tiene forma de U: demasiada memoria (d bajo) añade no-estacionariedad que
confunde al modelo; demasiada diferenciación (d alto) destruye la memoria útil.

### FFD en todos los horizontes (d=0.2)

| V_out | MAE FFD | MAE crudo | Δ | Usar FFD |
|-------|---------|-----------|---|---------|
| **1d** | **0.0112** | 0.0123 | **−8.9% ✓** | **SÍ** |
| 5d | 0.0057 | 0.0056 | +1.8% | NO |
| 30d | 0.0035 | 0.0024 | **+45.8% ✗** | NO |
| 90d | 0.0022 | 0.0013 | **+69% ✗** | NO |

**Regla**: FFD(d=0.2) SOLO si `V_OUT=1`. Para V_out≥5 empeora de forma severa porque la
memoria fraccional del log-precio introduce ruido de largo plazo cuando el target es el
promedio de 30 o 90 días futuros.

### Uso en código

```python
# Si V_OUT=1: activar FFD
X_src, y_src = load_data('data/precios.csv', ffd_d=0.2)
# X_src → serie FFD sobre log-precios (para X)
# y_src → retornos crudos alineados (para y, el target)

# Si V_OUT >= 5: retornos crudos
X_src, y_src = load_data('data/precios.csv')
```

En ambos casos:
```python
X, _ = create_time_series_data(X_src, V_IN, V_OUT)
_, y = create_time_series_data(y_src, V_IN, V_OUT)
```

**Nota**: FFD elimina los primeros ~1458 días (ventana de pesos con threshold=1e-5). Con
datos históricos desde 1945, esto no representa un problema en la tarea original. Con
datasets más cortos (< 5 años) hay que evaluar si vale la pena.

---

## 6. Qué normalización destruye y por qué

Experimentos sobre V_in=30, LSTM(64) como modelo de referencia, comparado con baseline
crudo (MAE=0.0123 para V_out=1):

| Técnica | MAE test | Δ vs crudo | Veredicto |
|---------|---------|-----------|----------|
| **Retornos crudos** | **0.0123** | — | BASELINE |
| StandardScaler (global) | 0.0128 | +4.1% | ✗ EMPEORA |
| Rolling Z-score (por ventana) | 0.0126 | +2.4% | ✗ EMPEORA |
| Feature Eng. (vol+mom+corr) | 0.0125 | +1.6% | ✗ EMPEORA |
| **FFD(d=0.2)** | **0.0112** | **−8.9%** | ✓ ÚNICO QUE MEJORA |

**Por qué empeora StandardScaler**: normalizar a media=0/std=1 con estadísticos globales
de décadas aplana todos los retornos a la misma escala. El modelo recibe el mismo input
tanto para el crash del 2008 como para un día tranquilo de 2015. La escala absoluta
contiene información sobre el régimen de volatilidad actual y no debe eliminarse.

**Por qué empeora Rolling Z-score**: normalizar por la propia ventana de entrada elimina
la magnitud absoluta por diseño. Dos ventanas idénticas en forma pero distintas en escala
se vuelven indistinguibles. El contexto de magnitud (¿estamos en un mercado de alta o baja
volatilidad?) desaparece.

**Por qué empeoran las features derivadas**: volatilidad realizada, momentum y correlación
cross-activo tienen estructura autocorrelada (GARCH para la vol, anomalía momentum) pero
el LSTM genérico sin mecanismo de atención no sabe priorizar qué features son relevantes
para predecir el nivel futuro bajo MAE. El modelo acaba usando el promedio de las 115
features derivadas, que no es mejor que las 23 originales.

**Regla práctica**: no normalizar sin testear empíricamente sobre val. La teoría general
dice "normaliza siempre para convergencia"; en datos financieros hay que verificarlo porque
la información de escala importa.

---

## 7. Preprocesado general para datasets nuevos

Si el hackathon entrega datos distintos a los de la tarea:

1. **Verificar formato**: ¿fechas como índice? ¿columnas de precios o retornos?
   ```python
   X_src, y_src = load_data('data/archivo.csv',
       price_cols=['Close', ...],   # si son precios
       return_cols=['ret_1', ...],  # si ya son retornos
       ffd_d=0.2 if V_OUT==1 else None)
   ```

2. **Inspeccionar antes de transformar**:
   ```python
   print(X_src.describe())
   print(X_src.isna().sum())
   X_src.plot(figsize=(14,3), alpha=0.3)
   ```

3. **Outliers en finanzas**: los eventos extremos (crisis) son información, no ruido.
   No eliminarlos automáticamente — el modelo debe aprender que existen.

4. **Missing values**: nunca hacer backward fill (introduce look-ahead bias: el valor
   de hoy "rellenando" ayer). Usar forward fill o eliminar la observación.

5. **Target desconocido**: si el target es una sola columna (no multivariante), ajustar:
   ```python
   n_targets = 1  # en build_model y train_ensemble
   ```

6. **Decisión de normalización**: probar primero sin normalización. Si el modelo no
   converge (curvas de train erráticas), considerar StandardScaler — pero verificar en
   val antes de aplicar al test final.
