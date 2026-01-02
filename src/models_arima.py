from statsmodels.tsa.arima.model import ARIMA
from pmdarima import auto_arima
import numpy as np
import pandas as pd


def arima_parameters(data):
    """
    Используется для одноразового подбора порядка ARIMA на обучающей 
    выборке. Параметры далее фиксируются.
    """
    model_auto = auto_arima(
        data,
        seasonal=False,
        stepwise=True,
        information_criterion='aic',
        trace=True,
        error_action='ignore',
        suppress_warnings=True
    )

    return model_auto.order


def prepare_arima_data(train_series):
    """
    Выполняет логарифмирование и взятие первых разностей.
    Предполагается, что порядок дифференцирования d = 1 фиксирован.
    Parameters
    ----------
    train_series : pandas.Series
        Исходный временной ряд для преобразования.
        Должен иметь числовой тип данных и индекс datetime.

    Returns
    -------
    train_log_diff : pandas.Series
        Логарифмированный ряд после взятия первых разностей.
    last_log : float
        Последнее значение логарифмированного исходного ряда.
    """
    train_log = np.log(train_series)
    train_log_diff = train_log.diff().dropna()
    last_log = train_log.iloc[-1]
    return train_log_diff, last_log


def fit_and_forecast_arima(train_log_diff, order, alpha=0.05):
    """
    Обучение модели ARIMA.

    Parameters
    ----------
    train_log_diff : pandas.Series
        Стационарный временной ряд (после логарифмирования и взятия разностей).
    order : tuple
        Порядок модели ARIMA (p, d, q).
    alpha : float, optional
        Уровень значимости для доверительного интервала (по умолчанию 0.05).

    Returns
    -------
    diff_pred : float
        Точечный прогноз для разностного ряда.
    ci_lower : float
        Нижняя граница доверительного интервала.
    ci_upper : float
        Верхняя граница доверительного интервала.
    """
    model = ARIMA(train_log_diff, order=order)
    results = model.fit()

    forecast = results.get_forecast(steps=1)
    diff_pred = forecast.predicted_mean.iloc[0]
    ci = forecast.conf_int(alpha=alpha).iloc[0]

    return diff_pred, ci.iloc[0], ci.iloc[1]


def inverse_transform(last_log, diff_pred, ci_lower, ci_upper):
    """
    Преобразование прогнозов разностного ряда в исходную шкалу.

    Parameters
    ----------
    last_log : float
        Последнее значение логарифмированного исходного ряда.
    diff_pred : float
        Прогноз для разностного ряда (логарифмированная разность).
    ci_lower : float
        Нижняя граница доверительного интервала для разностного ряда.
    ci_upper : float
        Верхняя граница доверительного интервала для разностного ряда.

    Returns
    -------
    pred : float
        Точечный прогноз в исходной шкале.
    lower : float
        Нижняя граница доверительного интервала в исходной шкале.
    upper : float
        Верхняя граница доверительного интервала в исходной шкале.
    """
    pred_log = last_log + diff_pred
    lower_log = last_log + ci_lower
    upper_log = last_log + ci_upper

    pred = np.exp(pred_log)
    lower = np.exp(lower_log)
    upper = np.exp(upper_log)

    return pred, lower, upper


def train_arima(full_series, test_series, arima_order):
    """
    Реализация walk-forward прогнозирования временного ряда с ARIMA и расширяющимся окном.а.

    Parameters
    ----------
    full_series : pandas.Series
        Полный временной ряд (обучающая и тестовая части).
    test_series : pandas.Series
        Тестовая часть ряда для прогнозирования.
    arima_order : tuple
        Порядок модели ARIMA (p, d, q).

    Returns
    -------
    results : pandas.DataFrame
        DataFrame с колонками:
        - forecast_arima: точечные прогнозы
        - lower_ci: нижняя граница доверительного интервала
        - upper_ci: верхняя граница доверительного интервала
        - residual: разница между фактическим значением и прогнозом(остатки)
        - actual: фактическое значение
    """
    preds, lowers, uppers, dates, residuals = [], [], [], [], []

    for test_date in test_series.index:
        # Подготовка данных
        train_series = full_series.loc[:test_date].iloc[:-1]
        train_log_diff, last_log = prepare_arima_data(train_series)

        # Прогноз
        diff_pred, ci_lower, ci_upper = fit_and_forecast_arima(
            train_log_diff, arima_order
        )

        # Обратное преобразование
        pred, lower, upper = inverse_transform(
            last_log, diff_pred, ci_lower, ci_upper
        )

        # Сбор результатов
        actual_value = full_series.loc[test_date]
        collect_results(
            test_date, actual_value, pred, lower, upper,
            preds, lowers, uppers, dates, residuals
        )

    results = pd.DataFrame({
        'forecast_arima': preds,
        'lower_ci': lowers,
        'upper_ci': uppers,
        'residual': residuals,
        'actual': test_series
    }, index=pd.to_datetime(dates))

    return results


def collect_results(date, actual, pred, lower, upper,
                    preds, lowers, uppers, dates, residuals):
    """
    Сбор результатов прогноза.
    Вынесено для уменьшения дублирования кода внутри цикла.
    """
    preds.append(float(pred))
    lowers.append(float(lower))
    uppers.append(float(upper))
    dates.append(date)
    residuals.append(float(actual - pred))
