from statsmodels.tsa.stattools import adfuller, kpss


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
