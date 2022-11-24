/*Data for the paper "..."

Aggregation (date_trunc)
------------------------
* Source data is collected in 5 minutes intervals, not much sense in using date_trunc < 5 min intervals
* Using mid_price for open and close, might be missing data (spikes) within the 5 min intervals.
* Alternatively use MAX(high), however that will give the max over past 24h (not within 5 min interval)

Resources
---------
* Get type descriptions here: https://docs.deribit.com/#public-get_book_summary_by_currency
*/

SELECT
	timestamp::date AS date,
	(array_agg(mid_price ORDER BY timestamp ASC))[1] AS mid_open,
	(array_agg(mid_price ORDER BY timestamp DESC))[1] AS mid_close,
	MAX(mid_price) as mid_high,
	MIN(mid_price) as mid_low,
	AVG(mid_price) as mid_average,
	AVG(mark_price) as mark_average,
	AVG(mid_price - mark_price) as mid_mark_avgdiff,
	AVG(volume) as volume,
	AVG(open_interest) as open_interest,
	instrument_name
FROM market.deribit

WHERE instrument_name in ('BTC-PERPETUAL', 'BTC-26MAR21') 
-- AND sectype = 'FUT'
-- AND base_currency = 'BTC'
AND timestamp::date <= '2021-03-26'
/*Option specific:*/
--AND pcflag = 'P'
--AND strike = 8000

GROUP BY
	date,
	instrument_name
	
ORDER BY date DESC

