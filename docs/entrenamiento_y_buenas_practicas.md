# Entrenamiento y Buenas Prácticas — Síntesis MIAX B3

Síntesis de: `training-nn-2026.md`, `b3_s4_mapa.md`, `redes-neuronales-fundamentos.md`.
Incluye reglas de la tarea previa, el reencuadre competitivo y la integración con
modelos fundacionales.

> **Nota**: este documento cubre la Capa 2 (custom NN desde cero). Para la Capa 1
> (zero-shot con Chronos-2/TimesFM) ver `docs/modelos_fundacionales.md`.

---

## 0. WORKAROUND GPU — LO PRIMERO

RTX 5070 Ti (Blackwell) es incompatible con TensorFlow GPU. **Antes de cualquier
import de TF/Keras**:

```python
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"   # CPU-only — si no se cuelga
import tensorflow as tf   # solo después
```

---

## 1. El loop de entrenamiento estándar

```python
# 1. Construir y compilar
model = build_model(tipo, V_IN, n_features=N_FEAT, n_targets=N_TARGETS,
                    units=64, dropout=0.0, lr=LR_MAP[tipo])

# 2. Entrenar (incluye restore del mejor epoch internamente)
hist = train_model(model, X_tr, y_tr, X_v, y_v, epochs=EPOCHS)

# 3. Evaluar en los tres splits
res = evaluate(model, X_tr, X_v, X_ts, y_tr, y_v, y_ts)
# res = {'train': 0.01xx, 'val': 0.01xx, 'test': 0.01xx, 'rmse': ..., 'dir': ...}
```

`train_model` internamente hace:
```python
model.fit(..., callbacks=[ReduceLROnPlateau, ModelCheckpoint], verbose=0)
model.load_weights(checkpoint_path)  # restaurar el mejor epoch al terminar
```

**Regla crítica**: siempre restaurar el mejor epoch antes de evaluar. El modelo al final
de `epochs` no es el mejor — puede haber sobreajustado en las últimas épocas.

---

## 2. Callbacks: la estrategia validada en la tarea

### Qué usar y qué no

| Callback | Usamos | Por qué |
|----------|--------|---------|
| `ReduceLROnPlateau` | ✓ Siempre | Reduce LR cuando el modelo se estanca, permitiendo refinamiento |
| `ModelCheckpoint` | ✓ Siempre | Guarda el mejor estado (menor val_loss) para restaurar al final |
| `EarlyStopping` | ✗ NO | Instrucción explícita del profesor y decisión de diseño propia |

### Por qué no EarlyStopping

Detener el entrenamiento antes de que termine produce dos problemas:
1. **Curvas incomparables entre modelos**: si el LSTM para en epoch 80 y el CNN1D en
   epoch 150, no puedes atribuir la diferencia de MAE a la arquitectura — puede ser
   simplemente que uno entrenó más.
2. **Oculta el comportamiento real**: queremos ver la curva completa para diagnosticar
   overfitting, plateau, o problemas de LR.

La combinación `ReduceLROnPlateau + ModelCheckpoint` logra lo mismo sin estos problemas:
el modelo entrena todas las épocas, el LR se ajusta cuando se estanca, y nos quedamos con
el mejor estado intermedio.

### Parámetros de los callbacks (nuestros valores)

```python
ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.9,       # reducción del 10% (más suave que el default 0.1)
    patience=15,      # esperar 15 épocas sin mejora antes de reducir
    min_lr=1e-5       # límite inferior para no "apagar" el aprendizaje
)
ModelCheckpoint(
    ckpt_path,
    monitor='val_loss',
    save_best_only=True
)
```

**Por qué factor=0.9 y no 0.5** (default): una reducción del 50% rompe el momentum de
Adam de golpe y puede desestabilizar el entrenamiento. Con 0.9 la reducción es gradual
y el modelo sigue explorando suavemente.

**Por qué patience=15**: con 300+ épocas y un plateau real, 10 épocas de paciencia son
pocas — se activa demasiado pronto en fluctuaciones normales. 15 épocas da suficiente
margen.

---

## 3. Diagnóstico por curvas de aprendizaje

`plot_history(hist)` muestra `train_loss` y `val_loss` por época. Saber leer la curva
es la habilidad más importante para diagnosticar y mejorar rápido:

| Patrón | Train loss | Val loss | Diagnóstico | Acción |
|--------|-----------|---------|-------------|--------|
| **Buen fitting** | Baja y estable | Baja, cercana a train | Óptimo | Ensemble, ajustar epochs |
| **Overfitting** | Muy baja | Alta y divergente | Exceso de capacidad | Dropout↑, L2↑, menos units |
| **Underfitting** | Alta | Alta, similar a train | Capacidad insuficiente | Más units, más épocas, LR↑ |
| **Sobrerregularizado** | Moderada | Mejor que train | Demasiado dropout/L2 | Reducir regularización |
| **Curvas erráticas** | Oscila fuerte | Oscila fuerte | LR incorrecto | Bajar LR, verificar datos |
| **Plateau prematuro** | Estancada rápido | Estancada rápido | LR muy alto | LR × 0.1, más épocas |

**Nuestro patrón observado en la tarea**: train ≈ val en todos los modelos, ambas cercanas
al mismo MAE (~0.0123 para V_out=1). Esto es **underfitting relativo** coherente con la
falta de señal en retornos. NO es un error de implementación — es la señal correcta de
que el problema es difícil. El foco debe ser reducir varianza, no reducir bias.

---

## 4. Regularización: cuándo aplicar qué

| Técnica | Cuándo | Cómo | Efecto |
|---------|--------|------|--------|
| **L2=1e-4** en Dense | Siempre en Dense(units) del MLP | `kernel_regularizer=l2(1e-4)` | Penaliza pesos grandes; suaviza la función |
| **Dropout(0.1–0.2)** | LSTM con >20K params o cuando val diverge de train | `LSTM(units, dropout=0.2)` | Ensemble implícito; reduce sobreajuste |
| **SpatialDropout1D(0.12–0.15)** | Salidas de Conv1D en modelos grandes (>20K) | `SpatialDropout1D(0.15)` | Apaga canales completos (más efectivo que Dropout para conv) |
| **Batch Normalization** | Redes >3 capas o con convergencia inestable | Después de capas ocultas | Normaliza activaciones; acelera convergencia |
| Sin regularización | Modelos pequeños (<10K params) | — | Los modelos pequeños no sobreajustan en este problema |

**Por qué SpatialDropout1D para convolucionales**: las salidas de Conv1D tienen alta
correlación entre posiciones del mismo filtro. Apagar el canal completo (SpatialDropout)
es mucho más disruptivo y efectivo que apagar neuronas individuales (Dropout normal).

**Observación de la tarea**: las 4 iteraciones de regularización en los modelos mixtos
(conv_bilstm, conv2_lstm) produjeron exactamente el mismo MAE en test. La regularización
mejoró las curvas de entrenamiento (convergencia más limpia) pero no el resultado final.
Esto refuerza que el cuello de botella es la señal, no el sobreajuste.

---

## 5. Hiperparámetros por modelo

### Learning rate (validado empíricamente)

Ver `docs/fundamentos_teoria.md` sección 4 para el razonamiento completo.

```python
LR_MAP = {
    'dense':    1e-4,   # a 3e-4 converge demasiado rápido (best epoch en epochs 3-28)
    'lstm':     3e-4,   # confirmado correcto en 96 experimentos
    'cnn1d':    3e-4,   # confirmado correcto
    'cnn_lstm': 3e-4,   # confirmado correcto
}
```

### EPOCHS recomendadas

| Modelo | EPOCHS base | EPOCHS para ensemble |
|--------|------------|---------------------|
| Dense | 50–100 | 100 |
| LSTM | 300 | 300 |
| CNN1D | 300–500 | 400 |
| CNN+LSTM | 300 | 300 |

Para el barrido inicial de 4 modelos en el hackathon: 150–200 épocas es suficiente para
detectar el ganador. Para el ensemble final: usar las EPOCHS completas.

### Unidades (units)

`units=64` es el default. Con datos distintos al de la tarea:
- Si el dataset es pequeño (<5K muestras train): reducir a `units=32`
- Si hay señal clara y recursos: probar `units=128`

---

## 6. Ensemble de semillas: la palanca principal

### Fundamento

El error de un ensemble de N modelos independientes en varianza es σ²/N (si son
incorrelados). Los distintos modelos entrenados en la misma arquitectura con distintas
semillas difieren solo en la inicialización aleatoria de pesos → son aproximadamente
incorrelados entre sí → el ensemble reduce la varianza sistemáticamente.

**Por qué es especialmente efectivo aquí**: las diferencias entre arquitecturas en nuestra
tarea son Δtest < 0.0001 en 15/16 combinaciones. El ruido de inicialización es del mismo
orden de magnitud que las diferencias reales. Un ensemble de 5 seeds "cancela" ese ruido
y deja la señal real.

### Uso práctico

```python
ens = train_ensemble(
    'lstm',          # mejor tipo del barrido
    X_tr, y_tr, X_v, y_v, X_ts, y_ts,
    V_in=V_IN, n_features=N_FEAT, n_targets=N_TARGETS,
    n_seeds=5,       # 3 si el tiempo aprieta, 10 si sobra
    epochs=EPOCHS,
    lr=LR_MAP['lstm']
)
print(f"MAE ensemble: {ens['mae_test']:.4f}")
print(f"Mejor seed:   {min(ens['seed_maes']):.4f}")
print(f"Media seeds:  {np.mean(ens['seed_maes']):.4f}")
```

### Ganancia esperada

| N seeds | Reducción de varianza típica | Coste temporal |
|---------|----------------------------|----------------|
| 3 | ~33–50% vs. semilla individual | 3× entrenamiento |
| **5** | **~50–65%** | 5× entrenamiento |
| 10 | ~70–80% | 10× entrenamiento |

El punto de rendimientos decrecientes está alrededor de 5–7 seeds. Con 4h disponibles,
5 seeds del mejor modelo es el equilibrio coste/beneficio óptimo.

**También se puede hacer ensemble de modelos distintos** (lstm + cnn1d + cnn_lstm). El
beneficio adicional es pequeño si los modelos tienen MAEs similares, pero suma algo de
diversidad.

---

## 7. Estrategia completa para el sábado

### Capa 1: modelos fundacionales (primero — objetivo < 30 min)

Ver `docs/modelos_fundacionales.md` para el código completo.

```
1. GPU workaround (os.environ antes de cualquier import)
2. Chronos-2 zero-shot → baseline multi-activo con cuantiles
3. TimesFM zero-shot → comparativa univariada
4. Comparar ambos con naive y media móvil
5. Si sobra tiempo: fine-tuning del mejor fundacional
```

### Capa 2: custom NN desde cero

La estrategia de dos extremos de Laparra (training-nn-2026):

```
Camino 1: Simple → Complejo (reducir bias, mejorar train)
  baseline lineal → dense pequeño → lstm pequeño → añadir capas/units

Camino 2: Complejo → Simple (reducir varianza, mejorar val)
  modelo grande que sobreajuste → reducir unidades → reducir capas → añadir dropout
```

En la práctica para 4h: ir directamente al barrido de los 4 modelos con defaults.
La búsqueda detallada de arquitectura es para proyectos de semanas.

**Secuencia óptima (Capa 2)**:
1. GPU workaround + ajustar `load_data`, verificar shapes (5 min)
2. Prueba rápida: LSTM 50 épocas para verificar pipeline (5 min)
3. Barrido 4 modelos × 150–200 épocas (15–30 min)
4. Ensemble 5 seeds del ganador × 300 épocas (30–60 min)
5. Si sobra: FFD si V_OUT=1, o aumentar seeds a 10

### Decisión final

Comparar MAE de todos los enfoques (fundacional zero-shot, fundacional fine-tuned,
custom NN simple, custom NN ensemble) y entregar el mejor. No hay jerarquía a priori:
el zero-shot puede ganar o puede perder según el problema concreto.

---

## 8. Resumen: lo que funciona vs lo que no

### Lo que funciona (validado empíricamente)

| Técnica | Impacto medido | Condición |
|---------|---------------|-----------|
| Ensemble 5 seeds | −0.0002 a −0.0005 MAE | Siempre |
| FFD(d=0.2) | −0.0011 MAE (−8.9%) | Solo V_out=1 |
| lr=1e-4 para MLP | Mejor exploración del espacio | Solo MLP/Dense |
| L2=1e-4 en Dense(64) | Mejora val en 15/16 combos | Solo capa Dense |
| batch_size=64 | Estabilidad sin sacrificar velocidad | Siempre |

### Lo que no funciona

| Técnica | Resultado | Por qué falla |
|---------|-----------|--------------|
| StandardScaler | +4.1% MAE | Pierde información de régimen de volatilidad |
| Rolling Z-score | +2.4% MAE | Elimina magnitud absoluta de retornos |
| Feature Engineering (vol+mom+corr) | +1.6% MAE | LSTM sin atención no prioriza features derivadas |
| Arquitectura multi-rama | ≈ igual | Cuello de botella es la señal, no la capacidad de separar |
| shuffle=True | Data leakage masivo | Viola el orden cronológico obligatorio |
| EarlyStopping | Curvas incomparables | Impide comparar modelos por número de épocas |

---

## 9. El reencuadre competitivo

Los 256 experimentos de la tarea mostraron convergencia a una zona estrecha. **El mensaje
correcto no es "da igual la arquitectura"** — es que el margen competitivo vive en el
epsilon y hay que reducirlo activamente.

**Los márgenes reales medidos**:
- Entre arquitecturas distintas: Δtest < 0.0001 en la mayoría de combinaciones (pero real y consistente)
- Ensemble 5 seeds vs mejor individual: ~−0.0003 a −0.0005 típico
- FFD(d=0.2) vs crudo para V_out=1: −0.0011

Esos números parecen pequeños pero en competición representan posiciones en el ranking.
La estrategia es acumular todos los Δ disponibles: arquitectura correcta + ensemble +
FFD si aplica + EPOCHS suficientes.
