from config import FRED_API_KEY
from fredapi import Fred


def load_gdp():
    """
    Загрузка данных реального ВВП (Real Gross Domestic Product) из базы данных FRED.
    
    Returns:
    --------
    pd.Series with DatetimeIndex
        Временной ряд реального ВВП (GDPC1) с датами в качестве индекса.
        
    Details:
    --------
    - Единицы измерения: Миллиарды долларов США в ценах 2017 года 
    - Частота данных: Квартальная
    - Период загрузки: С 1970-01-01 по 2024-12-31
    """
    fred = Fred(api_key=FRED_API_KEY)
    gdp_data = fred.get_series(
        'GDPC1', observation_start='1970-01-01', observation_end='2024-12-31')
    return gdp_data
