import numpy as np
import pandas as pd


def median_smooth_variable_edges(x, window):
    """
    Центрированная скользящая медиана с адаптивным окном на краях.
    
    Parameters:
    -----------
    x : pd.Series
        Входной временной ряд
    window : int
        Размер окна медианного фильтра (должен быть нечётным)
        
    Returns:
    --------
    pd.Series
        Сглаженный временной ряд той же длины, что и входной.
    """
    n = len(x)
    half = window // 2
    sm = np.empty(n)
    vals = x.values
    for i in range(n):
        left = max(0, i - half)
        right = min(n, i + half + 1)
        sm[i] = np.median(vals[left:right])
    return pd.Series(sm, index=x.index)


def apply_double_median_filter(series, window1=5, window2=3):
    """
    Применяет двойное медианное сглаживание и фильтр Хэмминга.
    Окна 5 и 3 используются в соответствии с классической процедурой 5–3 median filtering для оценки тренда временного ряда.
    
    Parameters
    ----------
    series : pd.Series
        Входной временной ряд
    window1 : int
        Размер первого окна медианного фильтра (нечётное)
    window2 : int
        Размер второго окна медианного фильтра (нечётное)
            
    Returns
    -------
        tuple : (сглаженный ряд, остатки)
    """
    # Медианное сглаживание: сначала 5-точечная, затем 3-точечная
    y_hat_5 = median_smooth_variable_edges(series, window=window1)
    y_hat_53 = median_smooth_variable_edges(y_hat_5, window=window2)
    
    # Фильтр Хэмминга (z_t = 0.25*y_{t-1} + 0.5*y_t + 0.25*y_{t+1})
    y_hat = 0.25 * y_hat_53.shift(1) + 0.5 * y_hat_53 + 0.25 * y_hat_53.shift(-1)
    
    y_hat.iloc[0] = y_hat_53.iloc[0]
    y_hat.iloc[-1] = y_hat_53.iloc[-1]
    
    residuals = series - y_hat
    
    return y_hat, residuals


def detect_outliers_mad(residuals, k=5.2):
    """
    Обнаружение выбросов на основе MAD (Median Absolute Deviation).
    
    Parameters:
    -----------
    residuals : pd.Series
        Остатки временного ряда
    k : float
        Коэффициент для определения порога выбросов
        
    Returns:
    --------
    tuple : (маска выбросов, mad, sigma_hat)
    """
    # Робастная оценка разброса: MAD и масштабированная sigma
    mad = np.median(np.abs(residuals - np.median(residuals)))
    sigma_hat_from_mad = 1.4826 * mad
    
    # Критерий: |r_t| > k * sigma_hat
    outlier_mask = residuals.abs() > (k * sigma_hat_from_mad)
    
    return outlier_mask, mad, sigma_hat_from_mad


def remove_outliers_by_trend(original_series, trend_series, outlier_mask):
    """
    Заменяет выбросы на значения оценки тренда.
    
    Parameters:
    -----------
    original_series : pd.Series
        Исходный временной ряд
    trend_series : pd.Series
        Оценка тренда
    outlier_mask : pd.Series
        Булева маска выбросов
        
    Returns:
    --------
    pd.Series : Очищенный временной ряд
    """
    cleaned_series = original_series.copy()
    cleaned_series[outlier_mask] = trend_series[outlier_mask]
    return cleaned_series