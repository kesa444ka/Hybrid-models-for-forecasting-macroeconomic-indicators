import numpy as np
import pandas as pd


def walk_forward_forecast_hybrid(
    arima_forecasts,
    test_series,
    train_residuals,
    lags,
    rf_model
):
    """
    Выполняет walk-forward прогнозирование гибридной модели (ARIMA + RF на остатках).

    Parameters
    ----------
    arima_forecasts : pd.Series
        Прогнозы ARIMA на тестовом периоде.
    test_series : pd.Series
        Фактические значения на тестовом периоде.
    train_residuals : pd.Series
        Остатки ARIMA на обучающей выборке (используются для инициализации лагов RF).
    lags : int
        Количество лагов, используемое в модели RF.
    rf_model : RandomForestRegressor
        Обученная модель RF для прогнозирования остатков.

    Returns
    -------
    pd.DataFrame
        DataFrame с колонками: 'actual', 'hybrid'.
    """
    residual_list = list(train_residuals)
    hybrid_preds = []
    
    for i in range(len(arima_forecasts)):
        # Прогноз ARIMA на текущем шаге
        arima_forecast = arima_forecasts.iloc[i]
        
        # Прогноз остатков с помощью RF
        last_lags = np.array(residual_list[-lags:]).reshape(1, -1)
        rf_forecast = rf_model.predict(last_lags)[0]
        
        # Гибридный прогноз
        hybrid_forecast = arima_forecast + rf_forecast
        hybrid_preds.append(hybrid_forecast)
        
        # Обновляем список остатков новым остатком
        actual_value = test_series.iloc[i]
        new_residual = actual_value - hybrid_forecast
        residual_list.append(new_residual)
    
    hybrid_results = pd.DataFrame({
        'actual': test_series.values,
        'hybrid': hybrid_preds
    }, index=arima_forecasts.index)
    
    return hybrid_results