Bitcoin Price from Coingecko and Bitcoin future price data from 29/05/2018 to 03/02/2021.
Data
- only include trading days of both assets
- Bitcoin price is the Coingecko hourly Bitcoin price closest to 2300 Berlin time
- Detail please check https://www.coingecko.com/en/methodology
- future price is the daily closing price (at 2300 Berlin time) from Bloomberg Terminal dated with Berlin Time
- sliced into train set with length of 300 and test set with length of 100.
- step size is 5
- in each set, testing data is the 100 trading days consecutive to the last training data

Tickers of Contracts included in the Bitcoin future data
- BTCH1 28/12/2020 - 03/02/2021
- BTCZ0 28/09/2020 - 24/12/2020
- BTCU0 29/06/2020 - 25/09/2020
- BTCM0 30/03/2020 - 27/03/2020
- BTCH0 30/12/2019 - 27/03/2020
- BTCZ19 30/09/2019 - 27/12/2019
- BTCU19 01/07/2019 - 27/09/2019
- BTCM19 01/04/2019 - 28/06/2019
- BTCH19 31/12/2018 - 29/03/2019
- BTCZ18 01/10/2018 - 28/12/2019
- BTCU18 02/07/2018 - 28/09/2018
- BTCM18 02/04/2018 - 29/06/2018
- BTCH18 01/01/2018 - 29/03/2018
