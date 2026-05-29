"""
utils.py — Funciones compartidas para el hackathon de predicción de retornos.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tempfile
from sklearn.model_selection import train_test_split
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint

_CKPT_PATH = tempfile.mktemp(suffix='.keras')

# ── CONSTANTES GLOBALES ───────────────────────────────────────────────────────
TICKERS = ['AEP','BA','CAT','CNP','CVX','DIS','DTE','ED','GD','GE',
           'HON','HPQ','IBM','IP','JNJ','KO','KR','MMM','MO','MRK','MSI','PG','XOM']

INPUT_WINDOWS  = [5, 10, 30, 90]   # días de entrada
OUTPUT_WINDOWS = [1,  5, 30, 90]   # días de salida
RANDOM_SEED    = 42
N_ASSETS       = 23

# ── CONSTANTES COMPARTIDAS DE DATOS ──────────────────────────────────────────
# ⚠️  CRÍTICO: estos valores se acuerdan entre los 3 miembros del equipo al
#    empezar el hackathon y NADIE los cambia en su notebook personal.
#    Cada uno experimenta solo con arquitectura e hiperparámetros de modelo.
#    Cambiar FILEPATH / V_IN / V_OUT / FFD_D invalida el ensemble del equipo
#    porque los splits ya no serán los mismos.
FILEPATH_SHARED    = 'data/precios.csv'
V_IN_SHARED        = 10
V_OUT_SHARED       = 1
FFD_D_SHARED       = None
PRICE_COLS_SHARED  = None   # None = todas las columnas numéricas como precios
RETURN_COLS_SHARED = None   # None = calcular retornos desde precios


# ── DATOS ─────────────────────────────────────────────────────────────────────
def create_time_series_data(data, V_in, V_out):
    """
    Genera pares (X, y) de ventanas deslizantes sobre la serie temporal.
    X : (N, V_in, n_features)  — ventana de entrada
    y : (N, n_features)        — promedio de los V_out pasos futuros
    Función del profesor — no modificar.
    """
    X, y = [], []
    arr = data.values if isinstance(data, pd.DataFrame) else data

    for i in range(len(arr) - V_in - V_out + 1):
        X.append(arr[i : i + V_in])
        if V_out > 0:
            y.append(np.mean(arr[i + V_in : i + V_in + V_out], axis=0))
        else:
            y.append(arr[i + V_in - 1])

    return np.array(X), np.array(y)


def make_splits(X, y, seed=RANDOM_SEED):
    """
    Partición en dos pasos, shuffle=False (orden cronológico obligatorio):
      Paso 1 → 90 % train_full / 10 % test
      Paso 2 → 75 % train    / 25 % val  (del train_full)
    Resultado: ~72 % train / ~18 % val / 10 % test
    """
    X_tr_full, X_ts, y_tr_full, y_ts = train_test_split(
        X, y, test_size=0.10, shuffle=False, random_state=seed)
    X_tr, X_v, y_tr, y_v = train_test_split(
        X_tr_full, y_tr_full, test_size=0.25, shuffle=False, random_state=seed)
    return X_tr, X_v, X_ts, y_tr, y_v, y_ts


# ── FFD (DIFERENCIACIÓN FRACCIONAL) ──────────────────────────────────────────
def apply_ffd(log_prices_df, d=0.2, threshold=1e-5):
    """
    Diferenciación Fraccional (López de Prado, cap. 5).
    d=0.2 es el óptimo medido experimentalmente: -8.9 % MAE vs retornos crudos.

    ADVERTENCIA: solo mejora para V_out=1 día.
                 Para V_out >= 5 empeora (hasta +45 %).

    Entrada : DataFrame de log-precios  →  np.log(precios)
    Salida  : DataFrame FFD (mismas columnas; primeros ~1 458 días eliminados)
    """
    arr = log_prices_df.values.astype(float)
    n, m = arr.shape
    w = [1.]
    for k in range(1, n):
        w.append(-w[-1] * (d - k + 1) / k)
        if abs(w[-1]) < threshold:
            break
    w = np.array(w[::-1])
    M = len(w)
    result = np.full((n, m), np.nan)
    for i in range(M - 1, n):
        result[i] = arr[i - M + 1 : i + 1].T @ w
    return pd.DataFrame(result, index=log_prices_df.index,
                        columns=log_prices_df.columns).dropna()


# ── CARGA DE DATOS ────────────────────────────────────────────────────────────
def load_data(filepath, price_cols=None, return_cols=None, ffd_d=None):
    """
    Cargador flexible para el hackathon.

    Parámetros
    ----------
    filepath    : ruta al CSV o parquet
    price_cols  : lista de columnas con precios → calcula log-retornos
                  None = usar todas las columnas numéricas como precios
    return_cols : lista de columnas ya en espacio de retornos (salta log-diff)
    ffd_d       : float → aplica FFD(d) sobre log-precios para X_src
                  Recomendado 0.2  |  Solo útil si V_OUT = 1

    Retorna
    -------
    X_src : datos para el input del modelo  (retornos o FFD)
    y_src : datos para el target y          (siempre retornos crudos)

    Uso en el notebook
    ------------------
    X_src, y_src = load_data('data/precios.csv')
    X, _ = create_time_series_data(X_src, V_IN, V_OUT)
    _, y = create_time_series_data(y_src, V_IN, V_OUT)
    """
    ext = os.path.splitext(filepath)[1].lower()
    df = (pd.read_parquet(filepath) if ext == '.parquet'
          else pd.read_csv(filepath, index_col=0, parse_dates=True))

    if return_cols is not None:
        returns   = df[return_cols].dropna()
        log_prices = None
    else:
        cols       = price_cols if price_cols is not None else df.select_dtypes(include=[np.number]).columns.tolist()
        prices     = df[cols].dropna()
        log_prices = np.log(prices)
        returns    = log_prices.diff().dropna()

    if ffd_d is not None:
        if log_prices is None:
            raise ValueError("ffd_d requiere columnas de precio (usa price_cols o no uses return_cols)")
        X_src = apply_ffd(log_prices, d=ffd_d)
        y_src = returns.loc[X_src.index]
        n_drop = len(returns) - len(X_src)
        print(f'FFD(d={ffd_d}): {len(X_src)} muestras  (eliminados {n_drop} días por ventana de pesos)')
    else:
        X_src = returns
        y_src = returns

    n_feat = X_src.shape[1]
    try:
        date_range = f'{X_src.index[0].date()} → {X_src.index[-1].date()}'
    except AttributeError:
        date_range = f'{X_src.index[0]} → {X_src.index[-1]}'
    print(f'Datos cargados: {X_src.shape[0]} filas × {n_feat} features  |  {date_range}')
    return X_src, y_src


# ── EVALUACIÓN ────────────────────────────────────────────────────────────────
def eval_mae(model, X, y):
    """MAE medio sobre todos los activos (escalar)."""
    return float(np.mean(np.abs(model.predict(X, verbose=0) - y)))


def eval_mae_naive(X, y):
    """MAE del naive forecast: predice el último retorno conocido."""
    return float(np.mean(np.abs(X[:, -1, :] - y)))


def eval_rmse(model, X, y):
    """RMSE sobre todos los activos."""
    return float(np.sqrt(np.mean((model.predict(X, verbose=0) - y) ** 2)))


def eval_directional(model, X, y):
    """Accuracy direccional: fracción de predicciones con el signo correcto."""
    y_pred = model.predict(X, verbose=0)
    return float(np.mean(np.sign(y_pred) == np.sign(y)))


# ── CALLBACKS ─────────────────────────────────────────────────────────────────
def get_callbacks(patience_lr=15, ckpt_path=None):
    """
    ReduceLROnPlateau + ModelCheckpoint sobre val_loss.
    Sin EarlyStopping: entrena todas las épocas para ver la curva completa.
    Llamar restore_best_weights(model) o model.load_weights(ckpt_path) tras fit().
    """
    if ckpt_path is None:
        ckpt_path = _CKPT_PATH
    return [
        ReduceLROnPlateau(monitor='val_loss', factor=0.9,
                          patience=patience_lr, min_lr=1e-5, verbose=0),
        ModelCheckpoint(ckpt_path, monitor='val_loss',
                        save_best_only=True, verbose=0),
    ]


def restore_best_weights(model):
    """Restaura los pesos del mejor epoch guardado por ModelCheckpoint."""
    model.load_weights(_CKPT_PATH)


def compile_model(model, lr=3e-4):
    """Compilación estándar: MAE + Adam. Misma para todos los modelos."""
    from keras.optimizers import Adam
    model.compile(loss='mean_absolute_error', optimizer=Adam(learning_rate=lr))
    return model


# ── CONSTRUCCIÓN DE MODELOS ───────────────────────────────────────────────────
def build_model(tipo, V_in, n_features=23, n_targets=None,
                units=64, dropout=0.0, lr=3e-4):
    """
    Construye y compila uno de los 4 modelos base.

    Parámetros
    ----------
    tipo       : 'dense' | 'lstm' | 'cnn1d' | 'cnn_lstm'
    V_in       : longitud de la ventana de entrada (días)
    n_features : número de features de entrada (activos)
    n_targets  : dimensión de salida  (default = n_features)
    units      : neuronas base para capas recurrentes y densas  (default 64)
    dropout    : tasa de dropout (solo lstm y cnn_lstm; 0.0 = sin dropout)
    lr         : learning rate  (dense recomendado: 1e-4; resto: 3e-4)

    Arquitecturas
    -------------
    dense    : Flatten → Dense(units, relu, L2=1e-4) → Dense(n_targets)
    lstm     : LSTM(units) [→ Dropout] → Dense(n_targets)
    cnn1d    : Conv1D(units,k=3)×2 → Conv1D(units//2,k=3) → GAP → Dense(units) → Dense(n_targets)
    cnn_lstm : Conv1D(units//2,k=3) → LSTM(units//2, dropout≥0.1) → Dense(n_targets)
    """
    from keras import Sequential, Input
    from keras.layers import (Dense, LSTM, Conv1D, GlobalAveragePooling1D,
                               Dropout, Flatten)
    from keras.regularizers import l2

    if n_targets is None:
        n_targets = n_features
    tipo = tipo.lower()

    if tipo in ('dense', 'mlp'):
        model = Sequential([
            Input((V_in, n_features)),
            Flatten(),
            Dense(units, activation='relu', kernel_regularizer=l2(1e-4)),
            Dense(n_targets),
        ])

    elif tipo == 'lstm':
        layers = [Input((V_in, n_features)), LSTM(units)]
        if dropout > 0:
            layers.append(Dropout(dropout))
        layers.append(Dense(n_targets))
        model = Sequential(layers)

    elif tipo in ('cnn1d', 'conv1d', 'conv'):
        model = Sequential([
            Input((V_in, n_features)),
            Conv1D(units,      kernel_size=3, activation='relu', padding='same'),
            Conv1D(units,      kernel_size=3, activation='relu', padding='same'),
            Conv1D(units // 2, kernel_size=3, activation='relu', padding='same'),
            GlobalAveragePooling1D(),
            Dense(units, activation='relu'),
            Dense(n_targets),
        ])

    elif tipo in ('cnn_lstm', 'conv_lstm'):
        model = Sequential([
            Input((V_in, n_features)),
            Conv1D(units // 2, kernel_size=3, activation='relu', padding='same'),
            LSTM(units // 2, dropout=max(dropout, 0.1)),
            Dense(n_targets),
        ])

    else:
        raise ValueError(
            f"tipo='{tipo}' desconocido. Opciones: 'dense', 'lstm', 'cnn1d', 'cnn_lstm'")

    return compile_model(model, lr=lr)


# ── ENTRENAMIENTO ─────────────────────────────────────────────────────────────
def train_model(model, X_tr, y_tr, X_v, y_v,
                epochs=300, batch_size=64, ckpt_path=None):
    """
    Entrena model con callbacks estándar y restaura los pesos del mejor epoch.
    Sin EarlyStopping.
    """
    hist = model.fit(
        X_tr, y_tr,
        validation_data=(X_v, y_v),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=get_callbacks(ckpt_path=ckpt_path),
        verbose=0,
    )
    if ckpt_path is not None:
        model.load_weights(ckpt_path)
    else:
        restore_best_weights(model)
    return hist


def train_ensemble(tipo, X_tr, y_tr, X_v, y_v, X_ts, y_ts,
                   V_in, n_features=23, n_targets=None,
                   n_seeds=5, epochs=300, batch_size=64, lr=3e-4, **kwargs):
    """
    Entrena n_seeds modelos con distintas semillas y promedia sus predicciones.

    PALANCA PRINCIPAL de reducción de varianza en competición.
    Las diferencias entre arquitecturas viven en el orden del ruido de
    inicialización; promediar semillas elimina ese ruido sistemáticamente.

    Retorna
    -------
    dict con claves:
      'y_pred_test' : predicciones del ensemble sobre test
      'y_pred_val'  : predicciones del ensemble sobre val
      'mae_test'    : MAE del ensemble en test
      'mae_val'     : MAE del ensemble en val
      'seed_maes'   : lista de MAE individuales por semilla
    """
    import tensorflow as tf

    if n_targets is None:
        n_targets = y_tr.shape[-1]

    preds_val, preds_test, seed_maes = [], [], []

    for seed in range(n_seeds):
        tf.random.set_seed(seed)
        np.random.seed(seed)
        ckpt = tempfile.mktemp(suffix=f'_seed{seed}.keras')
        model = build_model(tipo, V_in, n_features=n_features,
                            n_targets=n_targets, lr=lr, **kwargs)
        train_model(model, X_tr, y_tr, X_v, y_v,
                    epochs=epochs, batch_size=batch_size, ckpt_path=ckpt)
        preds_val.append(model.predict(X_v,  verbose=0))
        preds_test.append(model.predict(X_ts, verbose=0))
        mae = eval_mae(model, X_ts, y_ts)
        seed_maes.append(mae)
        print(f'  [seed {seed}]  MAE test = {mae:.4f}')

    y_ens_test = np.mean(preds_test, axis=0)
    y_ens_val  = np.mean(preds_val,  axis=0)
    mae_test   = float(np.mean(np.abs(y_ens_test - y_ts)))
    mae_val    = float(np.mean(np.abs(y_ens_val  - y_v)))

    print(f'\n  Ensemble ({n_seeds} seeds)  →  val={mae_val:.4f}  test={mae_test:.4f}')
    print(f'  Mejor semilla individual : {min(seed_maes):.4f}  '
          f'| Media semillas : {np.mean(seed_maes):.4f}')
    return {
        'y_pred_test': y_ens_test,
        'y_pred_val':  y_ens_val,
        'mae_test':    mae_test,
        'mae_val':     mae_val,
        'seed_maes':   seed_maes,
    }


def evaluate(model, X_tr, X_v, X_ts, y_tr, y_v, y_ts):
    """
    Evalúa el modelo en los tres splits.
    Retorna dict con MAE (train/val/test), RMSE test y accuracy direccional test.
    """
    return {
        'train':  eval_mae(model, X_tr, y_tr),
        'val':    eval_mae(model, X_v,  y_v),
        'test':   eval_mae(model, X_ts, y_ts),
        'rmse':   eval_rmse(model, X_ts, y_ts),
        'dir':    eval_directional(model, X_ts, y_ts),
        'params': model.count_params(),
    }


# ── RESULTADOS ────────────────────────────────────────────────────────────────
def build_results_df(results):
    """
    results : dict  { (modelo, V_in, V_out) : {'train', 'val', 'test', 'params'} }
    Devuelve un DataFrame con MultiIndex (modelo, V_in, V_out).
    """
    rows = []
    for (nombre, V_in, V_out), m in results.items():
        rows.append({'modelo': nombre, 'V_in': V_in, 'V_out': V_out,
                     'train': m['train'], 'val': m['val'], 'test': m['test'],
                     'params': m.get('params', 0)})
    df = pd.DataFrame(rows).set_index(['modelo', 'V_in', 'V_out'])
    return df


def best_per_window(results_df, metric='test'):
    """Matriz 4×4: mejor MAE en `metric` por (V_in, V_out)."""
    mat = np.full((4, 4), np.nan)
    for i, V_in in enumerate(INPUT_WINDOWS):
        for j, V_out in enumerate(OUTPUT_WINDOWS):
            subset = results_df.xs((V_in, V_out), level=('V_in', 'V_out'),
                                   drop_level=False)[metric]
            if not subset.empty:
                mat[i, j] = subset.min()
    return pd.DataFrame(mat, index=INPUT_WINDOWS, columns=OUTPUT_WINDOWS)


# ── VISUALIZACIÓN ─────────────────────────────────────────────────────────────
def plot_history(hist, title=''):
    """Curva loss / val_loss por época."""
    plt.figure(figsize=(6, 3))
    plt.plot(hist.history['loss'],     label='train')
    plt.plot(hist.history['val_loss'], label='val')
    plt.xlabel('Época'); plt.ylabel('MAE'); plt.title(title)
    plt.legend(); plt.tight_layout(); plt.show()


def plot_mae_matrix(mat_df, title='MAE en test'):
    """Heatmap seaborn 4×4 (filas=V_in, columnas=V_out)."""
    plt.figure(figsize=(6, 4))
    sns.heatmap(mat_df.astype(float), annot=True, fmt='.4f',
                cmap='YlOrRd_r', linewidths=.5)
    plt.xlabel('Ventana salida (días)'); plt.ylabel('Ventana entrada (días)')
    plt.title(title); plt.tight_layout(); plt.show()


def plot_model_comparison(results_df, V_in, V_out, metric='test'):
    """Barplot comparando MAE de todos los modelos para una combinación de ventanas."""
    subset = results_df.xs((V_in, V_out), level=('V_in', 'V_out'),
                           drop_level=False)[metric].reset_index(level=[1, 2], drop=True)
    ax = subset.plot(kind='bar', figsize=(7, 3), color='steelblue', edgecolor='k')
    ax.set_title(f'MAE test — entrada={V_in}d, salida={V_out}d')
    ax.set_ylabel('MAE'); ax.set_xlabel('Modelo')
    plt.xticks(rotation=30, ha='right'); plt.tight_layout(); plt.show()
