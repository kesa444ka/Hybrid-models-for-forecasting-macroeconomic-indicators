from statsmodels.tsa.stattools import adfuller, kpss
from sklearn.metrics import mean_absolute_error, mean_squared_error, confusion_matrix
import numpy as np


# ADF и KPSS тесты используются исключительно для предварительного анализа, а не для автоматического принятия решений в моделях
def adf_test_summary(timeseries):
    print('Результаты теста Дики-Фуллера:')
    print('=' * 50)

    adf_result = adfuller(timeseries, autolag='AIC')

    print(f'ADF Statistic: {adf_result[0]:.6f}')
    print(f'p-value: {adf_result[1]:.6f}')
    print(f'Количество наблюдений: {adf_result[3]}')
    print('Критические значения:')
    for key, value in adf_result[4].items():
        print(f'   {key}: {value:.3f}')

    return adf_result


def kpss_test_summary(timeseries):
    print('Результаты теста KPSS:')
    print('=' * 50)

    kpss_result = kpss(timeseries, regression='ct', nlags='auto')

    print(f'KPSS Statistic: {kpss_result[0]:.6f}')
    print(f'p-value: {kpss_result[1]:.6f}')
    print(f'Количество наблюдений: {len(timeseries)}')
    print('Критические значения:')
    for key, value in kpss_result[3].items():
        print(f'   {key}: {value:.3f}')

    return kpss_result


def calculate_metrics(y_true, y_pred):
    """
    Вычисление метрик качества прогноза в процентном выражении.

    Parameters
    ----------
    y_true : array-like
        Фактические значения.
    y_pred : array-like
        Прогнозируемые значения.

    Returns
    -------
    dict
        Словарь с метриками MAE, RMSE, MAPE в процентах.
    """
    
    mean_y = np.mean(y_true)
    return {
        "MAE, %": mean_absolute_error(y_true, y_pred) / mean_y * 100,
        "RMSE, %": np.sqrt(mean_squared_error(y_true, y_pred)) / mean_y * 100,
        "MAPE, %": np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    }


def calculate_DA_cm(actual_diff, pred_diff):
    """
    Вычисление статистики направленной точности (Directional Accuracy) и матрицы ошибок для знаков изменений.

    Parameters
    ----------
    actual_diff : pd.Series or array-like
        Фактивные изменения (разности) временного ряда.
    pred_diff : pd.Series or array-like
        Прогнозируемые изменения (разности) временного ряда.

    Returns
    -------
    DA_stat : float
        Статистика теста направленной точности (значимость правильности предсказания направления изменений).
    cm : ndarray
        Матрица ошибок для знаков изменений в формате:
        [[TP, FP],
         [FN, TN]]
        где:
        TP - правильно предсказанные положительные изменения,
        TN - правильно предсказанные отрицательные изменения,
        FP - неправильно предсказанные положительные изменения,
        FN - неправильно предсказанные отрицательные изменения.
    """
    x = pred_diff.values
    y = actual_diff.values

    # знаки изменений
    sign_x = np.sign(x)
    sign_y = np.sign(y)

    # выборочные средние знаков
    m_x = np.mean(sign_x)
    m_y = np.mean(sign_y)

    # оценка дисперсии
    V_DA = (1 - m_x**2) * (1 - m_y**2)

    # тестовая статистика DA
    T = len(sign_x)
    DA_stat = np.sqrt(T / V_DA) * (np.mean(sign_x * sign_y) - m_x * m_y)

    cm = confusion_matrix(sign_y, sign_x, labels=[1, -1])

    return DA_stat, cm