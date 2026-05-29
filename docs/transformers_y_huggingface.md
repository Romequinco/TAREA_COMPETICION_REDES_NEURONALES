# Transformers para Series Temporales y Búsqueda en HuggingFace Hub

Complemento a `modelos_fundacionales.md`. Cubre la arquitectura transformer
aplicada a TS, las variantes especializadas que han aparecido desde 2022,
y cómo usar HuggingFace Hub para encontrar y cargar modelos en minutos.

---

## 1. Por qué transformers en series temporales

El mecanismo de **atención** resuelve dos limitaciones históricas de LSTM/CNN:

| Problema | LSTM / CNN | Transformer |
|----------|-----------|-------------|
| Dependencias largas | Degradación de gradiente | Atención directa O(1) entre cualquier par de pasos |
| Paralelización | Secuencial → lento de entrenar | Totalmente paralelo → escala bien |
| Multivariado | Una serie cada vez o concatenación forzada | Atención cruzada entre variates de forma natural |

El precio: necesitan más datos y más cómputo que un LSTM para series cortas
(< 100 pasos). Por debajo de ese umbral, un LSTM o CNN 1D es habitualmente mejor.

---

## 2. Arquitecturas transformer especializadas en TS

Los transformers "vanilla" de NLP no funcionan bien directamente en TS porque:
- Los time steps son continuos, no tokens discretos
- La longitud de contexto suele ser grande (años de diario)
- Las series financieras tienen ruido alto y señal baja

Las variantes que sí funcionan:

### PatchTST (2023) — parches en lugar de pasos individuales

Divide la serie en **parches** (ventanas locales) y los trata como tokens.
Reduce la longitud de secuencia de N a N/patch_len → más eficiente y capta
tanto patrones locales como globales.

```
Serie:  [d1, d2, ..., d512]
Parche 1: [d1..d16]  → token 1
Parche 2: [d17..d32] → token 2
...
```

HF: `ibm/patchtst`  
Útil cuando: serie larga (>200 pasos), queremos capturar tendencias + ciclos.

### iTransformer (2024) — invierte la atención

En vez de aplicar atención a lo largo del tiempo, la aplica a través de las
**variates** (activos). Cada activo es un token; la atención capta correlaciones
entre activos. El tiempo se maneja con MLP en cada activo por separado.

```
Atención clásica: t1 ↔ t2 ↔ ... ↔ tN   (correlaciones temporales)
iTransformer:     activo1 ↔ activo2 ↔ ...  (correlaciones entre activos)
```

HF: `thuml/itransformer`  
Útil cuando: dataset multivariado donde las correlaciones entre activos importan.

### Autoformer / FEDformer / Informer

Variantes para **horizontes largos** (30–720 días) con complejidad O(N log N)
o O(N) en vez de O(N²). Menos relevantes para horizontes cortos (1–10 días).

### Foundation models (backbone transformer)

Los modelos de `modelos_fundacionales.md` (Chronos-2, TimesFM, Moirai, Kronos…)
son transformers preentrenados. La diferencia es que ya vienen entrenados: no
hay que diseñar ni entrenar la arquitectura. Ver ese doc para los detalles de uso.

---

## 3. HuggingFace Hub — cómo buscar modelos

### Búsqueda manual en la web

1. Ir a **huggingface.co/models**
2. Filtro **Task** → `time-series-forecasting`
3. Filtro **Libraries** → `transformers` / `pytorch` / `keras`
4. Ordenar por **Most Downloads** o **Trending**
5. Revisar la Model Card: buscar `financial`, `stock`, `returns`, `forecasting`

Tags útiles para buscar en la barra de búsqueda:
- `time-series`
- `forecasting`
- `financial`
- `stock-market`
- `chronos`, `timesfm`, `moirai`, `patchtst`, `itransformer`

### Búsqueda programática con `huggingface_hub`

```python
from huggingface_hub import list_models

# Todos los modelos de forecasting ordenados por descargas
modelos = list(list_models(
    task='time-series-forecasting',
    sort='downloads',
    direction=-1,       # descendente
    limit=20,
))
for m in modelos:
    print(m.modelId, ' | descargas:', m.downloads)
```

```python
# Buscar por keyword en el nombre
modelos = list(list_models(search='financial forecasting', limit=10))

# Ver los datasets con los que fue entrenado, métricas, etc.
from huggingface_hub import model_info
info = model_info('amazon/chronos-t5-base')
print(info.tags)
print(info.card_data)
```

### Evaluar si un modelo vale la pena (antes de descargarlo)

Revisar en la Model Card:
- **Datasets de preentrenamiento**: ¿incluye series financieras?
- **Horizonte soportado**: ¿cubre tu V_OUT?
- **Contexto máximo**: ¿acepta tu V_IN?
- **Multivariado**: ¿predice varios activos o solo uno?
- **Métricas reportadas**: ¿en qué benchmarks? ¿MAE o MASE?
- **Tamaño**: modelos > 1B params son difíciles de correr en CPU en 4 horas

---

## 4. Cargar y usar un modelo de HuggingFace

### Patrón general con `pipeline` (más fácil)

```python
from transformers import pipeline

# Si el modelo tiene pipeline de forecasting
forecaster = pipeline(
    'time-series-forecasting',
    model='nombre/del-modelo',
)
output = forecaster(series_historica, prediction_length=N_DIAS)
```

### Patrón con `AutoModel` (más control)

```python
from transformers import AutoConfig, AutoModel
import torch

config = AutoConfig.from_pretrained('nombre/del-modelo')
model  = AutoModel.from_pretrained('nombre/del-modelo')

# Preparar inputs según la Model Card del modelo concreto
inputs = {...}
with torch.no_grad():
    outputs = model(**inputs)
```

### Chronos-2 (ya con librería propia, la más cómoda)

```python
# pip install chronos-forecasting
from chronos import BaseChronosPipeline
import torch

pipeline = BaseChronosPipeline.from_pretrained(
    'amazon/chronos-t5-base',
    device_map='cpu',
    torch_dtype=torch.float32,
)
forecast = pipeline.predict(context, prediction_length=N_DIAS, num_samples=20)
mediana = forecast.median(dim=1).values
```

### PatchTST desde HF

```python
# pip install transformers
from transformers import PatchTSTConfig, PatchTSTForPrediction
import torch

config = PatchTSTConfig(
    num_input_channels=N_FEAT,
    context_length=V_IN,
    prediction_length=V_OUT,
    patch_length=16,
    stride=8,
)
model = PatchTSTForPrediction(config)
# Para zero-shot hay que usar un checkpoint preentrenado:
# model = PatchTSTForPrediction.from_pretrained('ibm/patchtst-etth1-pretrain')

inputs = {'past_values': torch.tensor(X_tr[:1], dtype=torch.float32)}
with torch.no_grad():
    out = model(**inputs)
pred = out.prediction_outputs  # (batch, prediction_length, n_feat)
```

---

## 5. FinBERT y modelos de texto financiero (distinto objetivo)

FinBERT y similares son modelos de **lenguaje** entrenados con texto financiero
(noticias, informes). Solo son útiles si el hackathon proporciona datos de texto
(titulares, sentiment) que quieras combinar con la serie de precios.

Para predicción de retornos a partir de precios históricos → usar los modelos
de series temporales, no los de texto.

---

## 6. Cuándo usar qué

| Situación | Recomendación |
|-----------|--------------|
| Primera prueba, sin tiempo | Zero-shot Chronos-2 (ver `modelos_fundacionales.md`) |
| Serie corta (< 100 pasos), pocos datos | LSTM o CNN 1D (`utils.py`) |
| Serie larga, muchas covariables | PatchTST o iTransformer |
| Varios activos correlacionados | iTransformer o Chronos-2 multivariado |
| Datos heterogéneos (precios + indicadores escalares) | Modelo funcional multi-rama (ver `modelos_fundacionales.md` §6) |
| Horizonte largo (> 30 días) | Autoformer / FEDformer / Informer |
| Dominio específicamente financiero | Kronos, FinCast (ver `modelos_fundacionales.md` §2) |
| Quiero fine-tuning eficiente | LoRA con PEFT (ver `modelos_fundacionales.md` §5) |

---

## 7. Flujo práctico para el hackathon

```
1. Buscar en HF: task=time-series-forecasting, ordenar por downloads
2. Leer la Model Card de los top-3: ¿soporta mi horizonte y contexto?
3. Zero-shot con Chronos-2 como baseline fundacional (< 10 min)
4. Si hay tiempo: probar PatchTST o iTransformer desde HF
5. Comparar con custom NN (utils.py) y con naive
6. Elegir el mejor por val MAE
```

---

## 8. Referencias

```
HuggingFace Hub (búsqueda):    huggingface.co/models?pipeline_tag=time-series-forecasting
huggingface_hub (librería):    pip install huggingface_hub
PatchTST (paper):              arxiv.org/abs/2211.14730
iTransformer (paper):          arxiv.org/abs/2310.06625
transformers (Hugging Face):   pip install transformers
Chronos-2:                     github.com/amazon-science/chronos-forecasting
KerasHub presets:              keras.io/keras_hub/presets/
Survey TSFM (2025):            arxiv.org/html/2504.04011v1
```
