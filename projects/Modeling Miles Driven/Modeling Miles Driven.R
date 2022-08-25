library("fpp2")

##Read the file in a dataframe and set it as a time series. 
miles=read.csv("totalmiles.csv")
colnames(miles)=c("date", "miles")

miles.ts=ts(miles$miles, start=c(1970, 1), frequency=12)

#Equalize by miles/day within each month/year so that comparisons can be
#meaningful and not skewed by month length/leap year status

miles_month.ts=monthdays(miles.ts)
miles.ts=miles.ts/miles_month.ts

autoplot(miles.ts)

#Estimate lambda for Box-Cox
lambda=BoxCox.lambda(miles.ts)
miles_box.ts=BoxCox(miles.ts, lambda)
autoplot(miles_box.ts)

#Let's try season plots and month plots
seasonplot(miles.ts, year.labels = TRUE, year.labels.left = TRUE, col=1:3)
monthplot(miles.ts)

#Cut the data smaller and check ACF/PACF

miles_clean=window(miles.ts, start=c(2010, 1), end=c(2020, 2))
autoplot(miles_clean)

Acf(diff(diff(miles_clean, 1, 1), 12, 1))
Pacf(diff(diff(miles_clean, 1, 1), 12, 1))

#Let's hit some ARIMAS

fit1=Arima(miles_clean, order=c(2,1,0), seasonal=c(0,1,0))
res=residuals(fit1)
Box.test(na.omit(res), lag=24, fitdf=2, type="L")
Acf(res)

fit2=Arima(miles_clean, order=c(2,1,0), seasonal=c(0,1,1))
res=residuals(fit2)
Box.test(na.omit(res), lag=24, fitdf=2, type="L")
Pacf(res)

fit3=auto.arima(miles_clean)
res=residuals(fit3)
Box.test(na.omit(res), lag=24, fitdf=2, type="L")
fit3

fit4 = hw(miles_clean, seasonal="additive", h=12)
res=residuals(fit4)
Box.test(na.omit(res), lag=24, fitdf=3, type="L")

fit5 = hw(miles_clean, seasonal="multiplicative", h=12)
res=residuals(fit5)
Box.test(na.omit(res), lag=24, fitdf=2, type="L")


autoplot(miles.ts) +
  autolayer(fitted(fit1), series = "order=c(2,1,0), seasonal=c(0,1,0)") +
  autolayer(fitted(fit2), series = "order=c(2,1,0), seasonal=c(0,1,1)") +
  autolayer(fitted(fit3), series = "Auto Arima") +
  autolayer(forecast(fit1, h=24)$mean, series = "order=c(2,1,0), seasonal=c(0,1,0)") +
  autolayer(forecast(fit2, h=24)$mean, series = "order=c(2,1,0), seasonal=c(0,1,1)") +
  autolayer(forecast(fit3, h=24)$mean, series="Auto Arima") +
  ggtitle("Forecast for daily miles") +
  xlab("Years") +
  ylab("Daily miles") +
  guides(colour = guide_legend(title = "Different Models"))
