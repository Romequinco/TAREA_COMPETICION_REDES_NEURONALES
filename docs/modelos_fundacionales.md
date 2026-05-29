# Modelos Fundacionales de Series Temporales — Guía para el Hackathon

Síntesis de la clase preparatoria (Laparra, 2026): Modelos Funcionales +
Modelos Fundacionales + Transfer Learning. Estrategia: **zero-shot primero,
custom NN como capa complementaria**.

---

## ⚠️ WORKAROUND GPU — LEER ANTES DE EMPEZAR

RTX 5070 Ti (arquitectura Blackwell) es incompatible con el soporte GPU de
TensorFlow. Esta línea **debe ir ANTES de cualquier import de TF/Keras** o el
proceso se cuelga indefinidamente:

```python
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # forzar CPU-only

# Solo después de esta línea:
import tensorflow as tf
import keras
```

Entorno de referencia (Daniel): Python 3.11.9 en `C:\venv_redes`.
Verificar si aplica en la máquina del hackathon antes de empezar.

---

## 1. Qué son y por qué usarlos primero

Un modelo fundacional de series temporales está preentrenado con enormes
volúmenes de series de naturalezas distintas (energía, tráfico, clima, finanzas,
salud...) esperando que las series compartan una "naturaleza propia" común. Cuando
le pasas TU serie, sabe trabajar con ella aunque nunca la haya visto.

Mecanismo: serie histórica → tokenización → Transformer → N días de predicción a
futuro. Es literalmente el problema del hackathon.

**Por qué empezar aquí**: zero-shot (sin entrenamiento propio) da un baseline en
minutos. Es gratis en tiempo de desarrollo y puede ser competitivo directamente.

El mismo salto ocurrido en visión por computador (ImageNet 2012 → modelos
reutilizables) está ocurriendo ahora en series temporales y datos financieros.

---

## 2. Catálogo de modelos

### Series temporales genéricas

| Modelo | Quién | Multivariado | Contexto max | Notas |
|--------|-------|-------------|-------------|-------|
| **Chronos-2** | Amazon (2025) | ✓ + covariables | — | **MÁS COMPLETO** para multi-activo |
| TimesFM 1.0 | Google (2024) | ✗ univariado | 512 | Simple; una serie cada vez |
| TimesFM 2.0 | Google (2025) | ✗ univariado | 2048 | Más contexto histórico |
| Timer-S1 | ByteDance/Tsinghua (2026) | ✓ | — | 8.3B params MoE; solo activa 750M |
| Reverso | MIT (2026) | — | — | Muy pequeño, buen error |
| Moirai | Salesforce | ✓ | — | También multivariado |

### Finance-specific (entrenados solo con datos financieros)

| Modelo | Params | Entrada | Salida | Notas |
|--------|--------|---------|--------|-------|
| **Kronos** | ~450M | 1024 | 128 | Multidimensional; tokeniza velas (K-line) |
| FinCast | ~185M | 768 | 96 | Sparse MoE; NO multidimensional |

El profesor: resultados financieros "un poco más esperanzadores" que los genéricos,
pero hay que verificarlo para el problema concreto.

**Regla multivariado**: para predecir varios activos a la vez → Chronos-2 o Moirai.
TimesFM hace una serie cada vez; para N activos necesitas N llamadas.

---

## 3. Formato de datos de entrada

La mayoría espera un DataFrame con columnas estándar:

```python
import pandas as pd

df = pd.DataFrame({
    'id':        ['AAPL', 'AAPL', 'GOOG', 'GOOG', ...],   # qué activo
    'timestamp': [pd.Timestamp('2024-01-01'), ...],          # fecha
    'target':    [150.2, 152.1, 140.0, 141.5, ...],         # lo que predices
    # covariables opcionales (indicadores, otros activos...):
    'cov_rsi':   [...],
    'cov_vol':   [...],
})
```

Para múltiples activos: apilar en un mismo DataFrame con `id` diferenciando cada
uno. El modelo predice N días hacia adelante para cada `id` y devuelve predicciones
**con cuantiles** (intervalos de confianza).

---

## 4. Zero-shot: el punto de partida obligatorio

*"Coges un modelo, le pasas las series y sacas las predicciones, ya está, a ver
qué pasa."* — Laparra (2026)

### Chronos-2 (recomendado para multi-activo)

```python
# pip install chronos-forecasting
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # GPU workaround PRIMERO

from chronos import BaseChronosPipeline
import torch
import pandas as pd

pipeline = BaseChronosPipeline.from_pretrained(
    "amazon/chronos-t5-base",    # empezar con el base; luego probar large
    device_map="cpu",
    torch_dtype=torch.float32,
)

# context: tensor (o lista de tensores) con la serie histórica
# Ejemplo con una serie:
context = torch.tensor(precios_historicos, dtype=torch.float32)

forecast = pipeline.predict(
    context=context,
    prediction_length=N_DIAS,
    num_samples=20,          # para obtener cuantiles
)
# forecast shape: (batch, num_samples, prediction_length)
mediana   = forecast.median(dim=1).values      # predicción central
q10, q90  = forecast.quantile(0.1, dim=1).values, forecast.quantile(0.9, dim=1).values
```

### TimesFM (univariado, más simple para empezar)

```python
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # GPU workaround PRIMERO

import timesfm

tfm = timesfm.TimesFm(
    hparams=timesfm.TimesFmHparams(
        backend="cpu",
        horizon_len=N_DIAS,
    ),
    checkpoint=timesfm.TimesFmCheckpoint(
        huggingface_repo_id="google/timesfm-1.0-200m"
    ),
)
tfm.initialize()

point_forecast, quantile_forecast = tfm.forecast_on_df(
    inputs=df,       # DataFrame con id, timestamp, target
    freq="D",        # frecuencia: "D" diaria
    value_name="target",
)
```

---

## 5. Niveles de transfer learning

Empezar siempre en Nivel 0. Subir solo si el resultado no es suficiente y hay tiempo.

| Nivel | Nombre | Descripción | Cuándo usar |
|-------|--------|-------------|-------------|
| **0** | **Zero-shot** | Usar el modelo sin tocar nada | Siempre primero — es gratis |
| 1 | Fine-tuning | Reentrenar con tus datos (entero o en parte) | Si zero-shot decepciona y tienes tiempo |
| 2 | Feature extractor | Quitar última capa; usar salida como features para tu propio regresor | Dominio muy distinto del preentrenamiento |
| 3 | Freeze + fine-tuning parcial | Congelar primeras capas; reentrenar solo el final | Pocos datos, problema parecido |

**Regla del cuadrante** (dos ejes: cantidad de datos × similitud con el preentrenamiento):

| | Pocos datos | Muchos datos |
|--|-------------|-------------|
| **Problema parecido** | Nivel 3: congelar casi todo | Nivel 1: descongelar bastante |
| **Problema distinto** | Difícil; probar Nivel 2 | Nivel 1: reentrenar entero |

Ejemplo del profesor: proyecto meteorología con la Agencia Europea. Tenían un modelo
que predecía TEMPERATURA → querían predecir HUMEDAD y OZONO. Congelaron la
extracción de características (ya "entiende" la atmósfera) y reentrenaron solo la
última capa. En UN DÍA tenían un modelo bastante bueno.

### Fine-tuning eficiente con LoRA (si se necesita)

En vez de reentrenar millones de pesos, se añade una corrección de bajo rango:

```
W_fine_tuned = W_preentrenado + ΔW = W_pt + A · B
```

`A` y `B` son matrices pequeñas (rango R). Solo se entrenan A y B (~0.26% de los
parámetros totales). Librería: **PEFT de Hugging Face**.

```python
from peft import get_peft_model, LoraConfig

config = LoraConfig(r=8, lora_alpha=32, target_modules=["query", "value"])
model  = get_peft_model(base_model, config)
# Entrenar model con tus datos...
```

---

## 6. Modelo funcional: mezclar series + indicadores escalares

Si el hackathon da precio histórico (secuencia) + indicadores económicos (escalares)
simultáneamente, la Functional API de Keras permite arquitecturas multi-rama:

```python
from keras import Input, Model
from keras.layers import Conv1D, GlobalAveragePooling1D, Dense, Concatenate

# Rama 1: serie temporal (V_in días, n_feat activos)
input_seq = Input(shape=(V_IN, N_FEAT), name='serie')
x = Conv1D(32, kernel_size=3, activation='relu', padding='same')(input_seq)
x = GlobalAveragePooling1D()(x)           # colapsa secuencia → vector de 32

# Rama 2: indicadores escalares (PMI, inflación, etc.)
input_esc = Input(shape=(N_ESCALARES,), name='indicadores')
y = Dense(8, activation='relu')(input_esc)   # → vector de 8

# Fusión: CONCATENAR (no sumar — dimensiones distintas)
merged = Concatenate()([x, y])    # → vector de 40
output = Dense(N_TARGETS)(merged)

model = Model(inputs=[input_seq, input_esc], outputs=output)
model.compile(loss='mae', optimizer='adam')

# Entrenamiento con múltiples entradas:
model.fit([X_serie, X_escalares], y_target, ...)
```

**Reglas clave del profesor**:

- **Concatenar vs sumar/restar**: sumar exige misma dimensión y mezcla la
  información; concatenar apila y deja que la red decida. Usa operaciones
  aritméticas solo cuando QUIERES esa semántica (ej. diferencia de dos señales).
- **No aplicar capas a un solo escalar**: las capas Dense combinan varios valores;
  aplicarlas a un escalar es darle vueltas a un número solo.
- **Escalado**: todas las entradas deben ir ~[0,1] o ~[-1,1]. Puedes dar más peso a
  un input usando un rango mayor, pero estás inyectando conocimiento a priori — si
  te equivocas, la red pierde tiempo deshaciendo esa elección.
- **Nombrar entradas y capas**; revisar `model.summary()` y `plot_model` antes de
  entrenar.
- **Desarrollar submodelos por separado** y combinarlos al final cuando la
  arquitectura sea compleja.

---

## 7. Avisos del profesor (críticos para el hackathon)

1. **Sesgo de reversión a la media**: estos modelos tienden a predecir
   comportamientos "normales" de serie. Si la serie subió fuerte, el modelo
   predice que bajará (reversión). NO tienen conocimiento financiero real; tienen
   memoria de lo que era "normal" en miles de series. Validar visualmente.

2. **Mira los cuantiles, no solo el valor central**: el intervalo de confianza es
   más informativo que la predicción puntual. Intervalo muy ancho = predicción poco
   fiable. Usar `num_samples` o los cuantiles del modelo para obtenerlos.

3. **Univariado vs multivariado**: para predecir varios activos a la vez usar
   Chronos-2 o Moirai. TimesFM hace una serie cada vez.

4. **Contexto y horizonte son máximos, no obligaciones**: puedes pasar menos
   contexto del máximo disponible y pedir menos días del máximo soportado.

5. **Datos sintéticos en preentrenamiento**: Chronos-2 y TimesFM-2 usan series
   Montecarlo. Buena generalización para patrones suaves/tendencias; pueden fallar
   con comportamientos muy peculiares de tu serie específica.

6. **No hay ganador universal**: probar varios modelos y comparar para TU problema
   concreto. *"Ponte y para tu problema haz la prueba."* — Laparra

---

## 8. Comparar siempre con baselines clásicos

El profesor insiste: comparar con métodos simples antes de celebrar victorias.

```python
# Baseline naive: última observación conocida
y_naive = X[:, -1, :]   # último timestep de la ventana de entrada

# Baseline media móvil
import pandas as pd
rolling_pred = df['target'].rolling(window=7).mean().shift(1)

# Baseline ARIMA (si hay tiempo)
from statsmodels.tsa.arima.model import ARIMA
# ...
```

Si el modelo fundacional no supera la media móvil, hay un problema de configuración
o el problema en cuestión tiene señal muy baja.

---

## 9. Referencias y repos

```
Chronos-2 (Amazon):  github.com/amazon-science/chronos-forecasting
                     HF: amazon/chronos-2
TimesFM 1.0 (Google):github.com/google-research/timesfm
                     HF: google/timesfm-1.0-200m
Timer-S1:            HF: bytedance-research/Timer-S1
Reverso:             HF: shinfxh/reverso
Kronos:              HF: NeoQuasar/Kronos-base
FinCast:             HF: Vincent05R/FinCast ; github.com/vincent05r/FinCast-fts
Survey TSFM (2025):  arxiv.org/html/2504.04011v1
KerasHub (Keras):    keras.io/keras_hub/presets/
PEFT (LoRA):         huggingface.co/docs/peft
```
