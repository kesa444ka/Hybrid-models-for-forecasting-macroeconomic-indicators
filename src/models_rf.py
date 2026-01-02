import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV


def create_lagged_features(series, n_lags):
    """
    Создание лаговые признаки для временного ряда.

    Parameters
    ----------
    series : array-like
        Временной ряд.
    n_lags : int
        Количество лагов для создания.

    Returns
    -------
    X : numpy.ndarray
        Матрица признаков с лагами.
    y : numpy.ndarray
        Целевые значения.
    """
    X, y = [], []
    for i in range(n_lags, len(series)):
        X.append(series[i-n_lags:i])
        y.append(series[i])
    return np.array(X), np.array(y)


def train_rf_model(
    X_train,
    y_train,
    param_grid,
    cv=3,
    scoring='neg_mean_squared_error',
    random_state=42
):
    """
    Подбор гиперпараметров и обучение модели Random Forest с использованием GridSearchCV.

    Parameters
    ----------
    X_train : numpy.ndarray
        Матрица признаков для обучения.
    y_train : numpy.ndarray
        Целевые значения.
    param_grid : dict
        Сетка гиперпараметров.
    cv : int
        Количество фолдов для кросс-валидации.
    scoring : str
        Метрика для оптимизации.
    random_state : int
        Seed для воспроизводимости.

    Returns
    -------
    best_model : RandomForestRegressor
        Лучшая модель после GridSearch.
    best_params : dict
        Лучшие параметры.
    """
    rf = RandomForestRegressor(random_state=random_state)
    grid_search = GridSearchCV(
        rf,
        param_grid,
        cv=cv,
        scoring=scoring,
        n_jobs=-1
    )
    grid_search.fit(X_train, y_train)

    return grid_search.best_estimator_, grid_search.best_params_


def prepare_rf_series(series):
    """
    Преобразование временного ряда для использования в модели Random Forest
    Выполнение логарифмирования и взятия первых разностей исходного ряда.

    Parameters
    ----------
    series : pandas.Series
        Исходный временной ряд.

    Returns
    -------
    log_diff_series : pandas.Series
        Стационарный ряд после логарифмирования и дифференцирования.
    last_log : float
        Последнее значение логарифмированного ряда(для обратного преобразования).
    """    
    log_series = np.log(series)
    log_diff_series = log_series.diff().dropna()
    last_log = log_series.iloc[-1]
    
    return log_diff_series, last_log


def fit_rf(log_diff_series, lags, model_params, random_state=42):
    """
    Обучение модели Random Forest на логарифмических разностях.

    Parameters
    ----------
    log_diff_series : pandas.Series
        Стационарный временной ряд после логарифмирования и дифференцирования.
    lags : int
        Количество лагов для создания признаков.
    model_params : dict
        Гиперпараметры для RandomForestRegressor.
    random_state : int, optional
        Seed для воспроизводимости результатов.

    Returns
    -------
    model: RandomForestRegressor
        Обученная модель.
    """    
    X_train, y_train = create_lagged_features(log_diff_series, lags)
    model = RandomForestRegressor(
        **model_params,
        random_state=random_state
    )
    model.fit(X_train, y_train)
    
    return model


def forecast_rf_one_step(model, log_diff_series, lags, last_log):
    """
    Одношаговый прогноз уровня временного ряда с использованием модели Random Forest.
    Прогнозируется значение логарифмической разности, после чего выполняется обратное преобразование к уровню.

    Parameters
    ----------
    model : RandomForestRegressor
        Обученная модель.
    log_diff_series : pandas.Series
        Логарифмические разности.
    lags : int
        Количество лагов.
    last_log : float
        Последнее значение логарифмированного исходного ряда.

    Returns
    -------
    float
        Прогноз на один шаг вперед в исходной шкале.
    """
    # Подготовка последних лагов
    last_lags = np.array(log_diff_series[-lags:]).reshape(1, -1)
    
    # Прогноз разности в логарифмической шкале
    diff_pred = model.predict(last_lags)[0]
    
    # Обратное преобразование
    pred_log = last_log + diff_pred
    pred = np.exp(pred_log)
    
    return float(pred)


def walk_forward_forecast_rf(
    full_series, 
    test_series, 
    lags, 
    model_params,
    random_state=42
):
    """
    Walk-forward (expanding window) прогнозирование временного ряда с использованием Random Forest.

    Parameters
    ----------
    full_series : pandas.Series
        Полный временной ряд.
    test_series : pandas.Series
        Тестовый временной ряд.
    lags : int
        Количество лагов.
    model_params : dict
        Гиперпараметры модели.
    random_state : int
        Seed для воспроизводимости.

    Returns
    -------
    results : pandas.DataFrame
        DataFrame с прогнозами и фактическими значениями.
    """
    preds = []
    pred_dates = []
    
    for test_date in test_series.index:
        # Подготовка данных для текущего окна
        train_series = full_series.loc[:test_date].iloc[:-1]
        train_log_diff, last_log = prepare_rf_series(train_series)
        
        # Обучение модели на текущем окне
        model = fit_rf(
            train_log_diff, lags, model_params, random_state
        )
        
        # Прогнозирование на один шаг
        pred = forecast_rf_one_step(model, train_log_diff, lags, last_log)
        
        # Сохранение результатов
        preds.append(pred)
        pred_dates.append(test_date)
    
    results = pd.DataFrame({
        'forecast': preds,
        'actual': test_series
    }, index=pd.to_datetime(pred_dates))
    
    return results


def get_feature_importance(model, lags):
    """
    Извлечение и сортировка важности признаков из модели Random Forest.

    Parameters
    ----------
    model : RandomForestRegressor
        Обученная модель Random Forest.
    lags : int
        Количество лагов (для названий признаков).

    Returns
    -------
    importance_df : pandas.DataFrame
        DataFrame с признаками и их важностью, отсортированный по убыванию.
    """
    importances = model.feature_importances_
    feature_names = [f'Lag {i}' for i in range(1, lags+1)]

    # Создание и сортировка DataFrame
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    })
    importance_df = importance_df.sort_values('importance', ascending=False)

    return importance_df
