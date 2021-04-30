CRIX_future_Open.csv 
Combined data of CRIX and CME future 
Columns:
date: UTC datetime 
CRIX: CRIX opening price of the day
future: CME Bitcoin future opening price of the day 
log return CRIX: log return of CRIX. log(CRIX_t/CRIX_t-1)
log return future: log return of CME Bitcoin future

CRIX is from crix.com download button. This is the up-to-date data source. The data presented on the website is outdated.
The CRIX is calculated from the opening price of a day timed in UTC. 

CME future data is downloaded from BBT. The timestamp is adjusted to Berlin timeby the BBT. We take the opening price of a day to match with the CRIX, by such, the time difference between the two prices is at most two hours.   
