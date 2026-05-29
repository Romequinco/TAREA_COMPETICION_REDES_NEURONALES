"""
utils.py — Hackathon MIAX 2026 — Predicción autorregresiva de 6 índices financieros.

Cambio central respecto a la tarea previa:
  - Problema : rollout autorregresivo 252 días (predecir día a día realimentando)
  - Métrica  : RMSE promedio sobre 6 índices (no MAE)
  - Estrategia: un modelo/baseline independiente por índice
"""

import os
import tempfile
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Checkpoint global (fallback cuando train_model no recibe ckpt_path)
_CKPT_PATH = tempfile.mktemp(suffix='.keras')


# ── CONSTANTES COMPARTIDAS ────────────────────────────────────────────────────
# ⚠️ CRÍTICO: se acuerdan UNA VEZ al inicio del hackathon.
# NADIE las cambia en su notebook personal — invalida la comparabilidad de backtests.

DATA_DIR    = 'data/'
N_DAYS_PRED = 252    # días a predecir en producción (el rollout completo)
VAL_DAYS    = 252    # días reservados como validación interna (= N_DAYS_PRED)
INDEX_COLS  = ['Index_A', 'Index_B', 'Index_C', 'Index_D', 'Index_E', 'Index_F']
RANDOM_SEED = 42

# ⚠️ PROVISIONAL — reacordar al ver los datos reales el día del hackathon.
# Si las series tienen < 1500 días hábiles, reducir a 10.
# Un V_IN grande con rollout de 252 días acumula más error de distribución.
# Cambiar solo si TODOS los miembros lo acuerdan ANTES de que cualquiera entrene.
V_IN_SHARED = 20


# ── CARGA ─────────────────────────────────────────────────────────────────────

def load_hackathon_data(data_dir=DATA_DIR):
    """
    Lee todos los CSV del hackathon disponibles en data_dir.
    Robusto a ficheros ausentes: avisa pero no falla.

    Retorna dict con las claves que existan:
      'train_indices'  : precios de cierre de los 6 índices (Index_A … Index_F)
      'test_dates'     : 252 fechas a predecir
      'train_news'     : titulares de noticias — train
      'test_news'      : titulares de noticias — test
      'train_macro'    : factores macro (oro, crudo, tipos) — train
      'test_macro'     : factores macro — test
      'train_network'  : métricas on-chain de Index_F — train
      'test_network'   : métricas on-chain de Index_F — test
    """
    files = {
        'train_indices': 'train_indices.csv',
        'test_dates':    'test_dates.csv',
        'train_news':    'train_news.csv',
        'test_news':     'test_news.csv',
        'train_macro':   'train_macro_factors.csv',
        'test_macro':    'test_macro_factors.csv',
        'train_network': 'train_network_metrics.csv',
        'test_network':  'test_network_metrics.csv',
    }
    data = {}
    for key, fname in files.items():
        path = os.path.join(data_dir, fname)
        if not os.path.exists(path):
            warnings.warn(f'[load_hackathon_data] No encontrado: {path}', stacklevel=2)
            continue
        try:
            df = pd.read_csv(path, index_col=0, parse_dates=True)
            data[key] = df
            try:
                rng = f'{df.index[0].date()} → {df.index[-1].date()}'
            except Exception:
                rng = f'{df.index[0]} → {df.index[-1]}'
            print(f'  [OK] {fname:<35} {df.shape[0]:>5} filas × {df.shape[1]} cols  ({rng})')
        except Exception as exc:
            warnings.warn(f'[load_hackathon_data] Error leyendo {path}: {exc}', stacklevel=2)

    if 'train_indices' in data:
        missing_idx = [c for c in INDEX_COLS if c not in data['train_indices'].columns]
        if missing_idx:
            warnings.warn(f'train_indices.csv no contiene: {missing_idx}', stacklevel=2)

    return data


# ── PREPROCESADO ──────────────────────────────────────────────────────────────

def make_window_dataset(series, v_in, use_log_rets=True, aux_features=None):
    """
    Ventanas deslizantes sobre una serie → (X, y).

    Parámetros
    ----------
    series       : 1D array-like (T,) — precios de un índice (o log-retornos si
                   use_log_rets=False y ya vienen convertidos)
    v_in         : int — días de historia en la ventana
    use_log_rets : bool (default True, RECOMENDADO).
                   True  → convierte series de precios a log-retornos antes de crear
                           las ventanas; y[i] = log-ret del día siguiente (~0.01,
                           escala agnóstica a la magnitud del índice).
                   False → trabaja con los valores de series sin conversión; útil si
                           ya pasas log-retornos calculados externamente, o si quieres
                           experimentar con precios crudos.
    aux_features : array-like (T, k) alineado con series (precios, no log-rets);
                   si use_log_rets=True, se ajusta automáticamente a la longitud de
                   los log-rets descartando la primera fila (convención: R[t] ~ aux[t])

    Retorna
    -------
    X : (N, v_in, n_features)  — n_features = 1 + k si hay aux, 1 si no
    y : (N,)                   — valor del día siguiente (log-ret o precio según modo)

    Nota: si use_log_rets=True, len(series) se reduce en 1 (np.diff).
    El modelo entrenado con use_log_rets=True debe usarse con backtest_autoregressive
    en log_ret_mode=True para que la reconstrucción a precios sea coherente.
    """
    arr = np.asarray(series, dtype=np.float32)

    if use_log_rets:
        arr = precios_a_logret(arr).astype(np.float32)  # longitud T-1

    main_feat = arr.reshape(-1, 1)

    if aux_features is not None:
        aux = np.asarray(aux_features, dtype=np.float32)
        if aux.ndim == 1:
            aux = aux.reshape(-1, 1)
        # Alinear longitud: si use_log_rets redujo arr en 1, tomar las últimas filas de aux
        if len(aux) > len(arr):
            aux = aux[-len(arr):]
        features = np.concatenate([main_feat, aux[:len(arr)]], axis=1)
    else:
        features = main_feat

    T = len(features)
    X_list, y_list = [], []
    for i in range(T - v_in):
        X_list.append(features[i : i + v_in])
        y_list.append(arr[i + v_in])  # log-ret o precio del día siguiente

    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.float32)


def make_temporal_split(series, val_days=VAL_DAYS, v_in=V_IN_SHARED):
    """
    Divide cronológicamente: los últimos val_days días son validación interna.

    Retorna
    -------
    series_train : todo lo anterior al período de validación
    series_val   : los últimos val_days días
    """
    arr = np.asarray(series, dtype=np.float32)
    if len(arr) < val_days + v_in:
        raise ValueError(
            f'Serie demasiado corta: {len(arr)} días. '
            f'Mínimo requerido: val_days + v_in = {val_days + v_in}.'
        )
    return arr[:-val_days], arr[-val_days:]


def apply_ffd(log_prices_df, d=0.2, threshold=1e-5):
    """
    Diferenciación Fraccional (López de Prado, cap. 5).
    d=0.2 fue óptimo en la tarea previa para V_out=1.
    Para rollout a 252 días su utilidad es incierta — verificar en backtest.
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


def precios_a_logret(series):
    """
    Convierte una serie de precios en log-retornos diarios.
    Salida de longitud len(series)-1: retorna[t] = log(P[t+1]/P[t]).
    Escala resultante ~0.01 — agnóstica a la magnitud absoluta del índice.
    """
    arr = np.asarray(series, dtype=np.float64)
    return np.diff(np.log(arr))


def logret_a_precios(log_rets, precio_inicial):
    """
    Reconstruye una serie de precios a partir de log-retornos y el precio de arranque.
    precio[t] = precio_inicial * exp(sum(log_rets[0..t]))

    precio_inicial debe ser el ÚLTIMO precio real conocido (antes del primer log-ret
    predicho). En el rollout nunca se usa ningún precio real del futuro — solo este
    punto de anclaje inicial.
    """
    rets = np.asarray(log_rets, dtype=np.float64)
    return float(precio_inicial) * np.exp(np.cumsum(rets))


def align_aux_features(train_indices, aux_df, feature_cols):
    """
    Alinea un DataFrame auxiliar (macro/network) con train_indices por fecha.
    Rellena huecos con forward-fill + backward-fill (festivos, datos tardíos).

    Retorna DataFrame con las mismas fechas que train_indices.
    """
    cols = [c for c in feature_cols if c in aux_df.columns]
    missing = set(feature_cols) - set(cols)
    if missing:
        warnings.warn(f'align_aux_features: columnas no encontradas: {missing}', stacklevel=2)
    return aux_df[cols].reindex(train_indices.index).ffill().bfill()


# ── MÉTRICAS ──────────────────────────────────────────────────────────────────

def rmse_per_index(y_true, y_pred):
    """
    RMSE por índice.
    Entradas 1D (una sola serie) → escalar.
    Entradas 2D (N, 6) → array de 6 valores.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    return np.sqrt(np.mean((y_true - y_pred) ** 2, axis=0))


def rmse_mean(y_true_dict, y_pred_dict):
    """
    RMSE medio entre los 6 índices — métrica oficial de la competición.

    Acepta:
      - dicts {index_name: array(252)}
      - arrays (252, 6) directamente
    """
    if isinstance(y_true_dict, dict):
        rmses = [float(rmse_per_index(y_true_dict[c], y_pred_dict[c]))
                 for c in INDEX_COLS if c in y_true_dict]
    else:
        rmses = list(rmse_per_index(y_true_dict, y_pred_dict).ravel())
    return float(np.mean(rmses))


def eval_directional(true_series, pred_series):
    """
    Accuracy direccional: fracción de días con signo de cambio correcto.
    Opera sobre series de precios (calcula signo del incremento diario).
    """
    true_arr = np.asarray(true_series, dtype=np.float64)
    pred_arr = np.asarray(pred_series, dtype=np.float64)
    return float(np.mean(np.sign(np.diff(true_arr)) == np.sign(np.diff(pred_arr))))


# ── BASELINES DE PRIMERA CLASE ─────────────────────────────────────────────────

def baseline_flat(series, n_steps):
    """
    Naive: repite el último precio conocido durante n_steps días.
    Baseline mínimo de referencia — 'aprobar' (RMSE < 75 000) probablemente
    requiere superarlo en los índices volátiles (A y F).
    """
    return np.full(n_steps, float(series[-1]), dtype=np.float64)


def baseline_drift(series, n_steps, window=20):
    """
    Proyecta la tendencia media de los últimos window días linealmente.
    drift = (precio[-1] − precio[-window]) / (window − 1)
    """
    arr = np.asarray(series, dtype=np.float64)
    recent = arr[-window:]
    daily_drift = (recent[-1] - recent[0]) / max(len(recent) - 1, 1)
    return np.array([arr[-1] + daily_drift * (i + 1) for i in range(n_steps)])


def baseline_random_walk(series, n_steps, n_sim=200, seed=RANDOM_SEED):
    """
    Ensemble de random walks con la volatilidad histórica de los últimos 60 días.
    Devuelve la mediana (más robusta que la media cuando hay deriva).
    """
    rng = np.random.default_rng(seed)
    arr = np.asarray(series, dtype=np.float64)
    window = min(61, len(arr))
    daily_rets = np.diff(arr[-window:]) / np.abs(arr[-window:-1] + 1e-10)
    mu    = np.mean(daily_rets)
    sigma = np.std(daily_rets)

    sims = np.empty((n_sim, n_steps), dtype=np.float64)
    for i in range(n_sim):
        shocks = rng.normal(mu, sigma, n_steps)
        sims[i] = arr[-1] * np.cumprod(1 + shocks)

    return np.median(sims, axis=0)


def eval_all_baselines(data_dict, val_days=VAL_DAYS, v_in=V_IN_SHARED):
    """
    Evalúa los 3 baselines sobre todos los índices en backtest autorregresivo.

    Retorna DataFrame: filas = baseline, columnas = índice + 'mean_rmse'.
    Ordenado de mejor a peor RMSE medio.
    """
    train_df = data_dict.get('train_indices')
    if train_df is None:
        raise ValueError("data_dict debe contener 'train_indices'")

    rows = {name: {} for name in ('flat', 'drift', 'random_walk')}

    for col in INDEX_COLS:
        if col not in train_df.columns:
            continue
        series = train_df[col].dropna().values
        train_s, val_s = make_temporal_split(series, val_days=val_days, v_in=v_in)
        preds = {
            'flat':        baseline_flat(train_s, val_days),
            'drift':       baseline_drift(train_s, val_days),
            'random_walk': baseline_random_walk(train_s, val_days),
        }
        for name, p in preds.items():
            rows[name][col] = float(rmse_per_index(val_s, p))

    df = pd.DataFrame(rows).T
    present = [c for c in INDEX_COLS if c in df.columns]
    df['mean_rmse'] = df[present].mean(axis=1)
    return df.sort_values('mean_rmse')


# ── ROLLOUT ────────────────────────────────────────────────────────────────────

def predict_autoregressive(predict_fn, ventana_inicial, n_steps,
                            precio_inicial=None, aux_data=None):
    """
    Bucle día-a-día que realimenta las predicciones como input del siguiente paso.
    Este es el núcleo del problema: los errores se acumulan en el rollout.

    Parámetros
    ----------
    predict_fn      : callable (1, v_in, n_features) → escalar
    ventana_inicial : array (v_in, n_features) — últimos v_in días conocidos
                      (log-retornos si el modelo opera en ese espacio; precios si no)
    n_steps         : int — días a predecir
    precio_inicial  : float opcional.
                      Si se pasa, se asume que el modelo predice log-retornos y esta
                      función reconstruye precios:
                        precio[t] = precio_inicial * exp(cumsum(log_rets_pred[0..t]))
                      Si es None, el modelo predice precios directamente y se devuelven
                      las predicciones crudas.
                      ⚠️ precio_inicial debe ser el ÚLTIMO precio real conocido del
                      período de entrenamiento. Nunca se usa ningún precio real del
                      período futuro — solo este punto de anclaje.
    aux_data        : array (n_steps, k) — features auxiliares de los días futuros
                      (p.ej. test_macro alineado con test_dates para Index_C/F)

    Retorna
    -------
    array (n_steps,): precios reconstruidos si precio_inicial es dado,
                      valores crudos (log-rets o precios) si no.

    Garantía de no-leakage: el bucle SIEMPRE alimenta la predicción propia del
    modelo en new_row[0]. Los valores reales del futuro no entran en ningún punto.
    """
    window = np.asarray(ventana_inicial, dtype=np.float32).copy()
    v_in, n_features = window.shape
    raw_preds = np.empty(n_steps, dtype=np.float32)

    for i in range(n_steps):
        x = window[np.newaxis, :, :]      # (1, v_in, n_features)
        next_val = float(predict_fn(x))   # log-ret predicho, o precio predicho
        raw_preds[i] = next_val

        # ── Construir nueva fila: SIEMPRE la predicción propia, nunca un valor real ──
        new_row = np.zeros(n_features, dtype=np.float32)
        new_row[0] = next_val             # feature principal: predicción realimentada
        if aux_data is not None and i < len(aux_data):
            aux_row = np.asarray(aux_data[i], dtype=np.float32).ravel()
            k = min(n_features - 1, len(aux_row))
            new_row[1 : 1 + k] = aux_row[:k]   # aux del futuro: conocido (test_macro, etc.)

        window = np.roll(window, -1, axis=0)
        window[-1] = new_row

    if precio_inicial is not None:
        # Reconstrucción: precio[t] = precio_inicial * exp(cumsum(log_rets))
        # raw_preds son log-retornos; precio_inicial es el último precio real conocido
        return logret_a_precios(raw_preds, precio_inicial).astype(np.float32)

    return raw_preds


# ── VALIDACIÓN INTERNA ─────────────────────────────────────────────────────────

def backtest_autoregressive(predict_fn, series, val_days=VAL_DAYS,
                            v_in=V_IN_SHARED, log_ret_mode=True, aux_data=None):
    """
    Simula el rollout autorregresivo completo reservando los últimos val_days días.

    ⚠️ CRÍTICO: único indicador fiable antes de quemar un intento de entrega.
    Un RMSE bajo en la ventana de entrenamiento NO predice el RMSE del rollout de
    252 días — solo este backtest lo hace. Ejecutar SIEMPRE antes de generar_submission.

    Parámetros
    ----------
    predict_fn   : callable (1, v_in, n_features) → escalar
    series       : 1D array COMPLETO de precios (el período de val está al final)
    val_days     : días reservados como validación (usar VAL_DAYS = 252)
    v_in         : ventana de entrada — debe coincidir con la del modelo entrenado
    log_ret_mode : bool (default True, RECOMENDADO).
                   True  → la ventana inicial y el rollout operan en log-retornos;
                           la reconstrucción a precios y el RMSE son en espacio de
                           PRECIOS (métrica oficial). El modelo debe haberse entrenado
                           con make_window_dataset(..., use_log_rets=True).
                   False → el modelo opera y predice precios directamente.
    aux_data     : array (T, k) alineado con series (misma longitud — se comprueba
                   con assert). Se extrae internamente el contexto y el período val.

    Retorna
    -------
    dict con:
      'rmse'         : RMSE del rollout en espacio de PRECIOS (métrica oficial)
      'preds'        : array (val_days,) predicciones de precios
      'true'         : array (val_days,) precios reales de validación
      'dir_accuracy' : accuracy direccional del rollout en espacio de precios

    Garantía de no-leakage:
      - La ventana inicial contiene solo datos del período de entrenamiento.
      - predict_autoregressive realimenta SIEMPRE sus propias predicciones.
      - val_s (precios reales) se usa ÚNICAMENTE al final para calcular RMSE,
        nunca como input al modelo durante el rollout.
    """
    arr = np.asarray(series, dtype=np.float32)

    # ── Validar alineación de aux_data ────────────────────────────────────────
    aux_arr     = None
    aux_context = None
    aux_val     = None
    n_features  = 1

    if aux_data is not None:
        aux_arr = np.asarray(aux_data, dtype=np.float32)
        if aux_arr.ndim == 1:
            aux_arr = aux_arr.reshape(-1, 1)
        assert len(aux_arr) == len(arr), (
            f"aux_data debe estar alineado con series (misma longitud): "
            f"aux_data={len(aux_arr)}, series={len(arr)}. "
            f"Un aux_data desalineado produce un backtest silenciosamente incorrecto."
        )
        aux_context = aux_arr[-(val_days + v_in) : -val_days]   # contexto de la ventana
        aux_val     = aux_arr[-val_days:]                         # período de validación
        n_features  = 1 + aux_arr.shape[1]

    # ── Split de precios ──────────────────────────────────────────────────────
    # val_prices: precios reales del período de validación — SOLO para medir RMSE al final
    train_prices, val_prices = make_temporal_split(arr, val_days=val_days, v_in=v_in)

    if log_ret_mode:
        # ── Modo log-retornos (default) ───────────────────────────────────────
        # Calcular log-rets sobre la serie COMPLETA (garantiza alineación perfecta)
        log_rets_full = precios_a_logret(arr).astype(np.float32)   # longitud T-1

        # Los últimos val_days log-rets corresponden al período de validación.
        # Los anteriores son el período de entrenamiento.
        train_rets = log_rets_full[:-val_days]   # NO incluye rets del período val
        # val_rets no se usa como input; solo se menciona aquí para claridad:
        # val_rets = log_rets_full[-val_days:]  ← nunca entra al modelo

        # Ventana inicial: últimos v_in log-rets del período de entrenamiento
        # ⚠️ Solo datos de train — sin ningún valor del futuro
        ventana = np.zeros((v_in, n_features), dtype=np.float32)
        ventana[:, 0] = train_rets[-v_in:]
        if aux_context is not None and len(aux_context) >= v_in:
            ventana[:, 1:] = aux_context[-v_in:]

        # Punto de anclaje para reconstrucción: último precio real conocido
        # Es el precio ANTERIOR al primer día del período de validación
        precio_inicial = float(train_prices[-1])

        # Rollout: el modelo predice log-rets; predict_autoregressive reconstruye precios
        preds_prices = predict_autoregressive(
            predict_fn, ventana, val_days,
            precio_inicial=precio_inicial,   # activa reconstrucción interna
            aux_data=aux_val
        )

    else:
        # ── Modo precio directo (backward compat / experimentos) ─────────────
        ventana = np.zeros((v_in, n_features), dtype=np.float32)
        ventana[:, 0] = train_prices[-v_in:]
        if aux_context is not None and len(aux_context) >= v_in:
            ventana[:, 1:] = aux_context[-v_in:]

        preds_prices = predict_autoregressive(
            predict_fn, ventana, val_days,
            precio_inicial=None,   # modelo predice precios directamente
            aux_data=aux_val
        )

    # ── Métrica final: SIEMPRE en espacio de precios ──────────────────────────
    rmse    = float(rmse_per_index(val_prices, preds_prices))
    dir_acc = eval_directional(val_prices, preds_prices)

    return {'rmse': rmse, 'preds': preds_prices, 'true': val_prices, 'dir_accuracy': dir_acc}


# ── DETECTIVE DE DATOS ────────────────────────────────────────────────────────

def lagged_correlation(df, target_col, candidate_cols=None, max_lag=30):
    """
    Correlación de target_col con cada columna candidata a lags 0..max_lag.

    Diseñada para cazar el Ghost (Index_D): pista dice que sigue con lag
    una señal oculta en otra columna. Buscar el lag donde la correlación es máxima.

    Parámetros
    ----------
    df             : DataFrame con todas las series (precios o retornos diarios)
    target_col     : columna objetivo (p.ej. 'Index_D')
    candidate_cols : columnas a probar; None = todas excepto target_col
    max_lag        : lag máximo en días

    Retorna
    -------
    DataFrame  índice = lag (0..max_lag), columnas = candidate_cols
    Valores    = coeficiente de correlación de Pearson
    Uso        : df_corr.abs().idxmax() → lag óptimo por columna candidata
    """
    if candidate_cols is None:
        candidate_cols = [c for c in df.columns if c != target_col]

    target = df[target_col].dropna()
    result = {}

    for col in candidate_cols:
        candidate = df[col].dropna()
        corrs = []
        for lag in range(max_lag + 1):
            if lag == 0:
                corr = target.corr(candidate)
            else:
                # candidate desplazado lag días atrás correlaciona con target actual
                aligned_target    = target.iloc[lag:].values
                aligned_candidate = candidate.iloc[:-lag].values
                min_len = min(len(aligned_target), len(aligned_candidate))
                corr = float(np.corrcoef(
                    aligned_target[:min_len],
                    aligned_candidate[:min_len]
                )[0, 1])
            corrs.append(corr)
        result[col] = corrs

    return pd.DataFrame(result, index=range(max_lag + 1))


# ── MODELOS NN ─────────────────────────────────────────────────────────────────

def compile_model(model, lr=3e-4, loss='mse'):
    """
    Compilación estándar: Adam + loss configurable.

    loss : 'mse' (default) — proxy directa para optimizar RMSE; gradientes estables
         | 'mae'           — más estable en series con picos extremos (Index_A, F)

    ⚠️ NO usar 'rmse' como loss: da el mismo óptimo que MSE (raíz monótona) pero
    con gradientes inestables cuando el error es pequeño (1/√ε → ∞).
    Regla: entrenar con MSE o MAE, reportar RMSE como métrica de evaluación.
    """
    from keras.optimizers import Adam
    model.compile(loss=loss, optimizer=Adam(learning_rate=lr))
    return model


def build_model(tipo, v_in, n_features=1, n_targets=1,
                units=64, dropout=0.0, lr=3e-4, loss='mse'):
    """
    Construye y compila uno de los 4 modelos base.

    tipo       : 'lstm' (DEFAULT recomendado — natural para rollout autorregresivo)
               | 'dense' | 'cnn1d' | 'cnn_lstm'
               Cambiar a otra arquitectura solo si el backtest_autoregressive
               muestra una mejora real en RMSE de rollout, no en loss de entrenamiento.
    v_in       : ventana de entrada (= V_IN_SHARED)
    n_features : 1 = solo precio; 1+k si se usan features auxiliares
    n_targets  : 1 por defecto (modelo por-índice)
    loss       : 'mse' (default) | 'mae' — ver compile_model para la razón
    """
    from keras import Sequential, Input
    from keras.layers import Dense, LSTM, Conv1D, GlobalAveragePooling1D, Dropout, Flatten
    from keras.regularizers import l2

    if n_targets is None:
        n_targets = n_features
    tipo = tipo.lower()

    if tipo in ('dense', 'mlp'):
        model = Sequential([
            Input((v_in, n_features)),
            Flatten(),
            Dense(units, activation='relu', kernel_regularizer=l2(1e-4)),
            Dense(n_targets),
        ])

    elif tipo == 'lstm':
        layers = [Input((v_in, n_features)), LSTM(units)]
        if dropout > 0:
            layers.append(Dropout(dropout))
        layers.append(Dense(n_targets))
        model = Sequential(layers)

    elif tipo in ('cnn1d', 'conv1d', 'conv'):
        model = Sequential([
            Input((v_in, n_features)),
            Conv1D(units,      kernel_size=3, activation='relu', padding='same'),
            Conv1D(units,      kernel_size=3, activation='relu', padding='same'),
            Conv1D(units // 2, kernel_size=3, activation='relu', padding='same'),
            GlobalAveragePooling1D(),
            Dense(units, activation='relu'),
            Dense(n_targets),
        ])

    elif tipo in ('cnn_lstm', 'conv_lstm'):
        model = Sequential([
            Input((v_in, n_features)),
            Conv1D(units // 2, kernel_size=3, activation='relu', padding='same'),
            LSTM(units // 2, dropout=max(dropout, 0.1)),
            Dense(n_targets),
        ])

    else:
        raise ValueError(
            f"tipo='{tipo}' desconocido. Opciones: 'dense', 'lstm', 'cnn1d', 'cnn_lstm'"
        )

    return compile_model(model, lr=lr, loss=loss)


# ── CALLBACKS ─────────────────────────────────────────────────────────────────

def get_callbacks(patience_lr=15, ckpt_path=None):
    """ReduceLROnPlateau + ModelCheckpoint sobre val_loss. Sin EarlyStopping."""
    from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
    if ckpt_path is None:
        ckpt_path = _CKPT_PATH
    return [
        ReduceLROnPlateau(monitor='val_loss', factor=0.9,
                          patience=patience_lr, min_lr=1e-5, verbose=0),
        ModelCheckpoint(ckpt_path, monitor='val_loss',
                        save_best_only=True, verbose=0),
    ]


def restore_best_weights(model, ckpt_path=None):
    """Restaura los pesos del mejor epoch guardado por ModelCheckpoint."""
    model.load_weights(ckpt_path or _CKPT_PATH)


# ── ENTRENAMIENTO ─────────────────────────────────────────────────────────────

def train_model(model, X_tr, y_tr, X_v, y_v,
                epochs=300, batch_size=64, ckpt_path=None):
    """Entrena con callbacks estándar y restaura los pesos del mejor epoch."""
    hist = model.fit(
        X_tr, y_tr,
        validation_data=(X_v, y_v),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=get_callbacks(ckpt_path=ckpt_path),
        verbose=0,
    )
    restore_best_weights(model, ckpt_path)
    return hist


def train_ensemble(tipo, X_tr, y_tr, X_v, y_v,
                   v_in, n_features=1, n_targets=1,
                   n_seeds=5, epochs=300, batch_size=64, lr=3e-4,
                   loss='mse', **kwargs):
    """
    Entrena n_seeds modelos y devuelve una predict_fn que promedia sus salidas.

    loss : 'mse' (default, optimiza RMSE) | 'mae' (más estable ante picos A/F)
           Comparar ambos con backtest_autoregressive; decide el RMSE del rollout,
           no la teoría.

    Retorna
    -------
    dict con:
      'predict_fn'  : callable (1, v_in, n_features) → escalar.
                      Pasar directamente a backtest_autoregressive o
                      predict_autoregressive para producción.
      'rmse_val'    : RMSE sobre X_v/y_v (ventana, NO rollout — solo orientativo)
      'seed_rmses'  : lista de RMSE individuales por semilla
      'models'      : lista de modelos Keras (para guardar a disco con .save())
    """
    import tensorflow as tf

    trained_models, preds_val, seed_rmses = [], [], []

    for seed in range(n_seeds):
        tf.random.set_seed(seed)
        np.random.seed(seed)
        ckpt = tempfile.mktemp(suffix=f'_seed{seed}.keras')
        model = build_model(tipo, v_in, n_features=n_features,
                            n_targets=n_targets, lr=lr, loss=loss, **kwargs)
        train_model(model, X_tr, y_tr, X_v, y_v,
                    epochs=epochs, batch_size=batch_size, ckpt_path=ckpt)
        p = model.predict(X_v, verbose=0).ravel()
        preds_val.append(p)
        seed_rmses.append(float(rmse_per_index(y_v, p)))
        trained_models.append(model)
        print(f'  [seed {seed}]  RMSE val (ventana) = {seed_rmses[-1]:.2f}')

    y_ens    = np.mean(preds_val, axis=0)
    rmse_ens = float(rmse_per_index(y_v, y_ens))
    print(f'\n  Ensemble ({n_seeds} seeds)  →  val RMSE (ventana) = {rmse_ens:.2f}')
    print('  ⚠️  Llamar backtest_autoregressive(result["predict_fn"], ...) para el RMSE real')

    def ensemble_predict_fn(x):
        return float(np.mean([m.predict(x, verbose=0).ravel()[0]
                               for m in trained_models]))

    return {
        'predict_fn': ensemble_predict_fn,
        'rmse_val':   rmse_ens,
        'seed_rmses': seed_rmses,
        'models':     trained_models,
    }


# ── ENTREGA ────────────────────────────────────────────────────────────────────

def generar_submission(predicciones_dict, test_dates, filepath='submission.csv'):
    """
    Construye el CSV de entrega 252 × 6.

    Parámetros
    ----------
    predicciones_dict : dict {index_name: array(252)} — predicciones de precios
    test_dates        : DataFrame o Series con las 252 fechas de test_dates.csv
    filepath          : ruta de salida

    Retorna el DataFrame generado (también lo guarda en filepath).
    """
    if isinstance(test_dates, pd.DataFrame):
        dates = pd.to_datetime(test_dates.iloc[:, 0])
    else:
        dates = pd.to_datetime(test_dates)

    df = pd.DataFrame(index=dates)
    df.index.name = 'Date'

    for col in INDEX_COLS:
        if col in predicciones_dict:
            df[col] = np.asarray(predicciones_dict[col], dtype=np.float64)
        else:
            warnings.warn(f'generar_submission: falta predicción para {col}', stacklevel=2)
            df[col] = np.nan

    df.to_csv(filepath)
    print(f'Submission guardada: {filepath}  ({df.shape[0]} filas × {df.shape[1]} cols)')
    return df


def validar_submission(filepath, test_dates=None):
    """
    Valida el CSV de entrega ANTES de subir — para no quemar un intento.
    Comprueba: existencia, dimensiones, columnas, fechas, NaN/Inf.

    Retorna True si pasa todo, False si hay algún error.
    Muestra detalle por índice (min/max/media/NaN) cuando es válido.
    """
    errors = []

    if not os.path.exists(filepath):
        print(f'[ERROR] Archivo no encontrado: {filepath}')
        return False

    df = pd.read_csv(filepath, index_col=0, parse_dates=True)

    if df.shape[0] != N_DAYS_PRED:
        errors.append(f'Filas: {df.shape[0]} (esperadas {N_DAYS_PRED})')
    if df.shape[1] != len(INDEX_COLS):
        errors.append(f'Columnas: {df.shape[1]} (esperadas {len(INDEX_COLS)})')

    missing_cols = set(INDEX_COLS) - set(df.columns)
    extra_cols   = set(df.columns) - set(INDEX_COLS)
    if missing_cols:
        errors.append(f'Columnas ausentes: {sorted(missing_cols)}')
    if extra_cols:
        errors.append(f'Columnas inesperadas: {sorted(extra_cols)}')

    nan_count = int(df.isnull().sum().sum())
    inf_count = int(np.isinf(df.select_dtypes(include=[np.number]).values).sum())
    if nan_count > 0:
        errors.append(f'NaN encontrados: {nan_count}')
    if inf_count > 0:
        errors.append(f'Inf encontrados: {inf_count}')

    if test_dates is not None:
        if isinstance(test_dates, pd.DataFrame):
            expected = pd.to_datetime(test_dates.iloc[:, 0])
        else:
            expected = pd.to_datetime(test_dates)
        if len(df) == len(expected):
            n_mismatch = int((df.index != expected.values).sum())
            if n_mismatch > 0:
                errors.append(f'{n_mismatch} fechas no coinciden con test_dates.csv')

    if errors:
        print('[FALLÓ] Submission NO válida:')
        for e in errors:
            print(f'  ✗ {e}')
        return False

    print('[OK] Submission válida:')
    print(f'     {df.shape[0]} filas × {df.shape[1]} columnas')
    try:
        print(f'     Rango: {df.index[0].date()} → {df.index[-1].date()}')
    except Exception:
        print(f'     Rango: {df.index[0]} → {df.index[-1]}')
    for col in INDEX_COLS:
        if col in df.columns:
            print(f'     {col}: min={df[col].min():.2f}  max={df[col].max():.2f}  '
                  f'media={df[col].mean():.2f}  NaN={df[col].isnull().sum()}')
    return True


# ── VISUALIZACIÓN ─────────────────────────────────────────────────────────────

def plot_history(hist, title=''):
    """Curva loss / val_loss por época."""
    plt.figure(figsize=(6, 3))
    plt.plot(hist.history['loss'],     label='train')
    plt.plot(hist.history['val_loss'], label='val')
    plt.xlabel('Época'); plt.ylabel('Loss'); plt.title(title)
    plt.legend(); plt.tight_layout(); plt.show()


def plot_rollout(true_series, pred_series, index_name='', val_days=VAL_DAYS,
                 show_context=120):
    """
    Visualiza el backtest autorregresivo: contexto histórico + predicción vs real.

    true_series  : array completo de precios (incluyendo el val period al final)
    pred_series  : array (val_days,) — predicciones del rollout
    show_context : días históricos a mostrar antes del corte de validación
    """
    arr   = np.asarray(true_series)
    preds = np.asarray(pred_series)
    n_train = len(arr) - val_days

    ctx_start = max(0, n_train - show_context)
    t_ctx = np.arange(ctx_start, n_train)
    t_val = np.arange(n_train, n_train + val_days)
    rmse_val = float(rmse_per_index(arr[-val_days:], preds))

    plt.figure(figsize=(13, 4))
    plt.plot(t_ctx, arr[ctx_start:n_train], color='steelblue', lw=0.9, label='histórico')
    plt.plot(t_val, arr[-val_days:],        color='seagreen',  lw=1.5, label='real (val)')
    plt.plot(t_val, preds,                  color='tomato',    lw=1.5, ls='--',
             label=f'predicho  RMSE={rmse_val:.0f}')
    plt.axvline(n_train, color='gray', ls=':', lw=1)
    plt.xlabel('Días'); plt.ylabel('Precio')
    plt.title(f'Backtest autorregresivo 252d — {index_name}')
    plt.legend(loc='upper left'); plt.tight_layout(); plt.show()


def plot_rmse_by_index(rmse_dict, title='RMSE por índice'):
    """
    Barplot de RMSE por índice para comparar estrategias.

    rmse_dict : {'Index_A': rmse, ...}                       → una estrategia
              | {'estrategia_1': {'Index_A': rmse, ...}, ...} → varias
    """
    first_val = next(iter(rmse_dict.values()))
    if not isinstance(first_val, dict):
        rmse_dict = {'resultado': rmse_dict}

    df = pd.DataFrame(rmse_dict).T.reindex(columns=INDEX_COLS)
    ax = df.plot(kind='bar', figsize=(10, 4), edgecolor='k')
    ax.set_title(title); ax.set_ylabel('RMSE'); ax.set_xlabel('Estrategia')
    plt.xticks(rotation=30, ha='right'); plt.tight_layout(); plt.show()
