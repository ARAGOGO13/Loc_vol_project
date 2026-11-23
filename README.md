# Arbitrage-free SPX volatility surface

Проект строит безарбитражные срезы опционных цен, поверхность полной подразумеваемой дисперсии и local volatility по формуле Дюпира.

Ноутбуки выполняются последовательно:

1. `00_data_preparation.ipynb`
2. `01_cubic_spline_baseline.ipynb`
3. `02_arbitrage_free_price_surface.ipynb`
4. `03_dupire_local_volatility.ipynb`

Промежуточные файлы создаются автоматически в `data/processed/`.