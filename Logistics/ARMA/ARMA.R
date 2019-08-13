data1<-read.csv('Total_Demand.csv');
library(forecast);
train<-data1[1:630,5];
test<-data1[632:701,5];


# off line training
sensor<-ts(train,frequency=7);
fit <- auto.arima(sensor,approximation=FALSE,trace=FALSE);
fcast <- forecast(fit,h=70,level=c(80,95));# typeof(fcast)
res <- data.frame(fcast[[4]],fcast[[5]],fcast[[6]]);#transfer time series back to data frame
write.csv(res, "ARMA_res.csv") 
# the five columns are point predictions, lower bound 80, lower bound 95, upper bound 80, upper bound 95






