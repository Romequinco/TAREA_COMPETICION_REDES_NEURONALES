# Fundamentos Teóricos — Síntesis MIAX B3

Síntesis de: `intro-deep-2026.md`, `b3_s1_mapa.md`, `calculo-optimizacion.md`.
Propósito: contexto teórico de arranque rápido para competición. Lo que contradijo
la teoría en la tarea previa se marca explícitamente.

---

## 1. El marco unificador: 4 componentes de cualquier sistema ML

Todo sistema de ML/DL es la combinación de exactamente estos 4 elementos:

| Componente | Qué define | Error de diseño frecuente |
|-----------|-----------|--------------------------|
| **Datos** {X, y} | La representación del problema | Normalizar antes de hacer el split temporal |
| **Modelo** f(X; θ) | La función que mapea entrada a salida | Escoger arquitectura sin testear la más simple |
| **Función de coste** L(ŷ, y) | QUÉ aprende el modelo | Usar MSE cuando el enunciado pide MAE |
| **Optimización** | CÓMO ajusta θ para minimizar L | LR fijo sin ReduceLROnPlateau |

*"La red es una versión comprimida de tus datos"* — Laparra (2026).
El modelo no puede extraer más de lo que los datos contienen. El diseño de la función de
coste y la calidad de los datos importan más que la arquitectura.

---

## 2. Riesgo de Bayes vs Riesgo Empírico

- **Riesgo de Bayes** R\*(f) = E[L(f(X), Y)]: el mínimo teórico inalcanzable.
  Requiere conocer la distribución real P(X,Y).
- **Riesgo Empírico** R̂(f) = (1/N) Σ L(f(Xᵢ), yᵢ): lo que minimizamos en la
  práctica con datos finitos.

La diferencia entre R̂ y R\* es el error de generalización. Reducirla es el objetivo de
la regularización y la validación correcta.

**Implicación directa para el hackathon**: si los datos no contienen señal directamente
explotable (como los log-retornos bajo EMH), el mínimo del riesgo empírico converge al
mejor estimador con esa distribución — la media bajo MAE. **Esto no significa que el
modelo no importe**: la diferencia entre converger exactamente a la media y desviarse
ligeramente en la dirección correcta ES el margen competitivo. El trabajo es encontrar y
ampliar ese Δ.

---

## 3. Función de coste: elegir bien define el objetivo

*"Es el objetivo de tu modelo"* — Laparra (2026). El modelo optimiza exactamente lo que
defines. Una función de coste mal elegida produce el modelo perfecto para el problema
equivocado.

### Tabla de funciones por tipo de problema

| Problema | Loss derivable (para entrenar) | Métrica (para evaluar) |
|----------|-------------------------------|------------------------|
| **Regresión** | **MAE** (robusto a outliers), MSE (penaliza errores grandes) | MAE, RMSE, R², **accuracy direccional** |
| Clasificación binaria | Binary cross-entropy | Accuracy, AUC, F1, Precision/Recall |
| Clasificación multiclase | Categorical cross-entropy | Accuracy, AUC |
| Segmentación | IoU (Jaccard), Dice | IoU |
| Series temporales probabilistas | Quantile loss, NLL | CRPS |

**Para el hackathon**: MAE como loss (el enunciado lo requiere; más robusto que MSE a los
retornos con colas pesadas). Reportar siempre **MAE + accuracy direccional** — la precisión
en la dirección del movimiento tiene valor financiero real.

**Distinción importante**: la loss debe ser derivable (backprop la necesita); la métrica
no. No usar accuracy directamente como loss en clasificación.

---

## 4. Optimizadores: default teórico vs lo que medimos

### Comparativa teórica

| Optimizador | Mecanismo | Cuándo usar |
|-------------|-----------|-------------|
| SGD + Nesterov | Gradiente con momentum anticipado | Regularización implícita; más robusto en producción |
| RMSprop | LR adaptativo por peso (normaliza por media cuadrática) | Fallback cuando Adam no converge |
| **Adam** | Momentum + LR adaptativo (Kingma & Ba, 2015) | **Primera opción en casi todo** |

**Regla teórica**: Adam con `lr=3e-4` (la "Karpathy constant"). Funciona en la gran
mayoría de problemas de regresión sin ajuste.

### Corrección empírica de la tarea previa

La tarea reveló una discrepancia real con el default teórico para modelos densos:

| Modelo | LR teórico | LR empírico | Evidencia |
|--------|-----------|-------------|-----------|
| LSTM, GRU, CNN y variantes | 3e-4 | **3e-4** (confirmado) | Convergencia correcta, best epoch distribuido en épocas 50-300 |
| **Dense / MLP** | 3e-4 | **1e-4** (mejor) | A 3e-4: best epoch en epochs 3-28 en 15/16 combinaciones. Δval < 0.0005 pero desaprovecha el entrenamiento |

**Por qué a 3e-4 el MLP converge demasiado rápido**: Adam acumula momentum en los
primeros pasos y con LR alto llega a un mínimo local plano muy pronto. A 1e-4 el
modelo explora más el espacio de parámetros antes de asentarse.

**Regla para el hackathon**: usar el `LR_MAP` de `entrenamiento.ipynb`:
```python
LR_MAP = {'dense': 1e-4, 'lstm': 3e-4, 'cnn1d': 3e-4, 'cnn_lstm': 3e-4}
```

---

## 5. Descenso por gradiente: variantes

Backpropagation aplica la regla de la cadena hacia atrás para calcular ∂L/∂θ en cada
capa. Es simplemente la derivada en cadena — una técnica matemática con 350 años de
antigüedad, aplicada a grafos computacionales.

| Variante | Batch size | Gradiente | Velocidad | Uso |
|----------|-----------|-----------|-----------|-----|
| Batch GD | N (todo) | Exacto | Muy lento | Problemas pequeños |
| **Mini-batch GD** | 32–256 | Aproximado | Rápido | **Standard** |
| SGD estocástico | 1 | Muy ruidoso | Máximo | Sólo con mucho ruido intencional |

**Usamos `batch_size=64`**: balance entre calidad del gradiente y velocidad. CPU-friendly.
El cuello de botella computacional en backprop es la **memoria RAM**, no la velocidad de
cómputo: la RAM limita el batch_size máximo porque hay que guardar todas las activaciones
intermedias para calcular los gradientes.

---

## 6. Arquitecturas: cuándo elegir cada tipo

### Input shape y preprocesado por familia

| Tipo | Input a Keras | Preprocesado | Preserva temporal |
|------|--------------|--------------|------------------|
| **Dense/MLP** | `(N, V_in × feat)` | `X.reshape(N, -1)` via Flatten | ✗ Pierde estructura |
| **LSTM** | `(N, V_in, feat)` | Directo | ✓ Completo |
| **CNN1D** | `(N, V_in, feat)` | Directo; `padding='same'` para V_in pequeños | ✓ Local |
| **CNN+LSTM** | `(N, V_in, feat)` | Directo | ✓ Local + global |

### Descripción de las 4 arquitecturas base

**Dense** (`build_model('dense', ...)`):
```
Flatten → Dense(units, relu, L2=1e-4) → Dense(n_targets)
```
Ventaja: rápido, estable. Desventaja: pierde completamente el orden temporal.
El V_in no aporta información más allá del primer paso (todo se aplana).

**LSTM** (`build_model('lstm', ...)`):
```
LSTM(units) [→ Dropout] → Dense(n_targets)
```
Gating mechanism (input/forget/output gates) para aprender qué recordar y qué olvidar.
Mejor ratio params/MAE histórico. Primera opción para series temporales.

**CNN1D** (`build_model('cnn1d', ...)`):
```
Conv1D(units,k=3)×2 → Conv1D(units//2,k=3) → GAP → Dense(units) → Dense(n_targets)
```
Detecta patrones locales de 3 días consecutivos. `padding='same'` para mantener longitud
de secuencia con V_in pequeños. GAP (GlobalAveragePooling) hace el modelo invariante a la
longitud de entrada.

**CNN+LSTM** (`build_model('cnn_lstm', ...)`):
```
Conv1D(units//2,k=3) → LSTM(units//2, dropout≥0.1) → Dense(n_targets)
```
Conv extrae patrones locales; LSTM integra la información a lo largo del tiempo.
El dropout mínimo de 0.1 es necesario para regularizar la LSTM en este tipo de
combinación (la conv ya habrá reducido algo la señal).

### Jerarquía de complejidad (Laparra)

Para aumentar capacidad: más neuronas → más capas → cambiar tipo de capa.
Para reducir overfitting: quitar neuronas → pooling → quitar capas.

Orden creciente: Dense < Recurrentes < Convolucionales < Mixtos.

---

## 7. No-linealidad y activaciones

Sin funciones de activación no lineales, múltiples capas Dense se colapsan en una sola
transformación lineal (inútil para aprender relaciones no lineales).

**ReLU** (f(x) = max(0,x)) domina desde 2012:
- Derivada simple (0 si x<0, 1 si x>0): sin gradiente desvaneciente
- Introduce sparsity natural (neuronas inactivas = 0)
- Computacionalmente trivial

**Para la capa de salida en regresión**: activación lineal (ninguna). Nunca sigmoid o
softmax en regresión — truncarían el rango de salida.

---

## 8. Modelos Funcionales (Keras Functional API)

El modelo Sequential apila capas en línea recta. La Functional API permite varias
entradas, varias salidas y ramas en paralelo. Es necesaria cuando los datos tienen
**dimensionalidades distintas** que no caben en la misma entrada.

Ejemplo del profesor (pizarra): predecir precio con datos de DOS tipos:
- Serie temporal de precios (forma `(20, 2)`) → entra por una rama con Conv1D o LSTM
- Indicador económico mensual (escalar) → entra por otra rama con Dense

```python
from keras import Input, Model
from keras.layers import Conv1D, GlobalAveragePooling1D, Dense, Concatenate

input_seq = Input(shape=(V_IN, N_FEAT), name='serie')
x = Conv1D(32, kernel_size=3, activation='relu', padding='same')(input_seq)
x = GlobalAveragePooling1D()(x)          # colapsa secuencia → vector fijo

input_esc = Input(shape=(N_ESCALARES,), name='indicadores')
y = Dense(8, activation='relu')(input_esc)

merged = Concatenate()([x, y])           # CONCATENAR, no sumar (dimensiones distintas)
output = Dense(N_TARGETS)(merged)

model = Model(inputs=[input_seq, input_esc], outputs=output)
model.fit([X_serie, X_escalares], y_target, ...)  # una X por cada entrada
```

**Regla clave**: Concatenar (apilar) vs sumar/restar (mezclar).
Usar sumar/restar solo cuando QUIERES esa operación semántica (ej. diferencia entre
dos señales). Concatenar es la opción por defecto cuando solo quieres combinar info.

Para guía completa de arquitecturas multi-rama y modelos fundacionales → ver
`docs/modelos_fundacionales.md`.

---

## 9. Hardware: el cuello de botella real

- La **memoria RAM** es el límite, no la velocidad de procesamiento
- Backprop necesita mantener todas las activaciones intermedias en memoria
- CPU con 8 GB RAM: batch_size máximo ≈ 64–128 para estos modelos
- **RTX 5070 Ti (Blackwell) incompatible con TF GPU** → usar workaround CPU-only
  (ver `docs/modelos_fundacionales.md`, sección 0)

Para reducir uso de memoria: reducir batch_size, simplificar arquitectura.
