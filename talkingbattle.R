train.data <- fread("gender_age_train.csv",colClasses = c("character","factor","integer","factor"))
setkey(train.data,"device_id")
test.data <- fread("gender_age_test.csv",colClasses = c("character"))
setkey(test.data,"device_id")
test.data[,':='(gender=NA,age=NA,group=NA)]
total.data <- rbind(train.data,test.data)
setkey(total.data,device_id)
train.devices <- unique(train.data$device_id)
test.devices <- unique(test.data$device_id)
rm(train.data,test.data)
gc()

brand.info <- fread("phone_brand_device_model.csv",colClasses = c("character","character","character"),encoding = "UTF-8")
brand.info <- unique(brand.info)
setkey(brand.info,"device_id")
brand.info[,device_model := paste0(phone_brand," ",device_model)]
brand.info <- unique(brand.info)

total.info <- merge(total.data,brand.info,all.x = TRUE)
rm(total.data)
gc()


events.info <- fread("events.csv",colClasses = c("character","character","character","numeric","numeric"))
events.info[,timestamp:=parse_datetime(timestamp,"%Y-%m-%d %H:%M:%S")]
setkeyv(events.info,c("event_id","device_id"))
app.events.info <- fread("app_events.csv",colClasses = c("character","character","integer","integer"))
setkeyv(app.events.info,c("event_id","app_id"))
app.labels.info <- fread("app_labels.csv",colClasses = c("character","character"))
app.labels.info <- unique(app.labels.info)
setkeyv(app.labels.info,c("app_id","label_id"))
label.categories.info <- fread("label_categories.csv",colClasses = c("character","character"))
setkey(label.categories.info,"label_id")

events.apps.info <- merge(events.info,app.events.info,by="event_id",all.x=TRUE,allow.cartesian = TRUE)
events.apps.info[,':='(timestamp=NULL,longitude=NULL,latitude=NULL)]
rm(app.events.info,app.labels.info,events.info,label.categories.info)
gc()

devices.with.app.info <- events.apps.info[,.(length=.N,NAs=sum(is.na(app_id))),by=.(device_id,app_id)][length>1 & NAs>0,device_id]
events.apps.info <- events.apps.info[!(device_id%in%devices.with.app.info&is.na(app_id))]
gc()

devices.with.noapp.info <- events.apps.info[is.na(app_id),device_id]

events.apps.summary <- events.apps.info[,.(activity=sum(is_active,na.rm=TRUE)/sum(is_installed,na.rm = TRUE)),by=.(device_id,app_id)]
rm(events.apps.info,devices.with.app.info)
gc()

final.merge <- merge(events.apps.summary,total.info,by="device_id",all.y=TRUE)
setkeyv(final.merge,c("device_id","app_id"))
final.merge[!is.na(app_id),app_id:=paste0("App ID:",app_id)]
final.merge[,':='(phone_brand=paste0("Brand:",phone_brand),device_model=paste0("Model:",device_model))]


unique_devices <- final.merge[,device_id] %>% unique()
unique_mapps <- final.merge[,app_id] %>% unique()
unique_apps <- final.merge[!is.na(app_id),app_id] %>% unique()
unique_brands <- final.merge[,phone_brand] %>% unique()
unique_models <- final.merge[,device_model] %>% unique()

id_devices <- match(final.merge[,device_id],unique_devices)
id_mapps <- match(final.merge[,app_id],unique_mapps)
id_apps <- match(final.merge[,app_id],unique_apps)
id_brands <- match(final.merge[,phone_brand],unique_brands)
id_models <- match(final.merge[,device_model],unique_models)

id_devices_apps <- cbind(id_devices,id_apps) %>% unique()
id_devices_apps <- id_devices_apps[complete.cases(id_devices_apps),]
id_devices_mapps <- cbind(id_devices,id_mapps) %>% unique()
id_devices_brands <- cbind(id_devices,id_brands) %>% unique()
id_devices_models <- cbind(id_devices,id_models) %>% unique()

unique_mapps[is.na(unique_mapps)] <- "Missing Apps"

device_app.matrix <- Matrix(0,nrow=length(unique_devices),ncol=length(unique_mapps),dimnames=list(unique_devices,unique_mapps),sparse = TRUE)
device_app.matrix[id_devices_mapps] <- 1
device_app_activity.matrix <- Matrix(0,nrow=length(unique_devices),ncol=length(unique_apps),dimnames=list(unique_devices,unique_apps),sparse = TRUE)
device_app_activity.matrix[id_devices_apps] <- final.merge[!is.na(app_id),activity]

device_brand.matrix <- Matrix(0,nrow=length(unique_devices),ncol=length(unique_brands),dimnames=list(unique_devices,unique_brands),sparse = TRUE)
device_brand.matrix[id_devices_brands] <- 1

device_models.matrix <- Matrix(0,nrow=length(unique_devices),ncol=length(unique_models),dimnames=list(unique_devices,unique_models),sparse = TRUE)
device_models.matrix[id_devices_models] <- 1

X_full <- cbind(device_brand.matrix,device_models.matrix,device_app.matrix,device_app_activity.matrix)
X_full <-cbind(device_brand.matrix,device_models.matrix,device_app.matrix)
Y_full <- total.info[rownames(X_full),group]
Y_full <- factor(Y_full)

X_train <- X_full[!is.na(Y_full),]
X_test <- X_full[is.na(Y_full),]

Y_train <- Y_full[!is.na(Y_full)]
Y_test <- Y_full[is.na(Y_full)]


ftrain <- xgb.DMatrix(X_train,label=as.integer(Y_train)-1,missing=NA)


depth <- 8
shrk <- 0.01
ntree <- 300


param <- list(booster="gblinear",
              num_class=length(unique(Y_train)),
              objective="multi:softprob",
              eval_metric="mlogloss",
              eta=shrk,
              max.depth=depth,
              lambda=5,
              alpha=2,
              subsample=0.5,
              colsample_bytree=0.5,
              num_parallel_tree=2,
              num_thread=2)


watchlist <- list(train=ftrain)

set.seed(1812)

fit_cv <- xgb.cv(params=param,
                 data=ftrain,
                 nrounds=1000,
                 watchlist=watchlist,
                 nfold=5,
                 early.stop.round=3,
                 verbose=1)
# 
# fit.xgb <- xgb.train(params=param,
#                      data=ftrain,
#                      nrounds=ntree,
#                      watchlist=watchlist,
#                      verbose=1)
# 
# ftest <- xgb.DMatrix(X_test,label=as.integer(Y_test),missing=NA)
# 
# 
# pred <- predict(fit.xgb,ftest)
# pred <- t(matrix(pred,nrow=length(levels(Y_full))))
# colnames(pred) <- levels(Y_full)
# first_submit <- cbind(device_id=rownames(X_test),as.data.frame(pred))
# write.csv(first_submit,file="submit0.2.csv",row.names=F,quote=F)

search.grid <- expand.grid(
  max_depth = c(4,6,8),
  eta = c(0.01,0.025),
  nrounds = 1000,
  lambda = c(4,5,6),
  alpha=c(1,2,3)
)


# param <- list(booster="gblinear",
#               num_class=length(unique(Y_train)),
#               objective="multi:softprob",
#               eval_metric="mlogloss",
#               subsample=0.5,
#               colsample_bytree=0.5,
#               num_parallel_tree=3)
# 
# xgbLinearCtrl <- trainControl(method="repeatedcv",repeats=5,number=5,summaryFunction = mnLogLoss)
# xgbLinearModel <- train(x=X_train,y=as.integer(Y_train)-1,
#   method="xgboost",
#   maximize = FALSE,
#   trControl = xgbLinearCtrl,
#   tuneGrid = search.grid,
#   params=param
# )


fit_cv <- xgb.cv(params=param, data=ftrain, nrounds=100000, watchlist=watchlist, nfold=5, early.stop.round=3, verbose=1)



get_cv_mlogloss <- function(x){
  start.time <- Sys.time()
  param <- list(booster="gblinear",
                num_class=length(unique(Y_train)),
                objective="multi:softprob",
                eval_metric="mlogloss",
                eta=search.grid[x,"eta"],
                max.depth=search.grid[x,"max_depth"],
                lambda=search.grid[x,"lambda"],
                alpha=search.grid[x,"alpha"],
                subsample=0.5,
                colsample_bytree=0.5,
                num_parallel_tree=1,
                num_thread=1,
                nthread=2)
  print(start.time)
  print(x)
  print(search.grid[x,])
  fit_cv <- xgb.cv(params=param, data=ftrain, nrounds=search.grid[x,"nrounds"], watchlist=watchlist, nfold=5, early.stop.round=25, verbose=1,print.every.n = 5)
  min_tree <- which(fit_cv$test.mlogloss.mean==min(fit_cv$test.mlogloss.mean),arr.ind = TRUE)
  end.time <- Sys.time()
  print(end.time)
  print(end.time-start.time)
  return(cbind(index=x,min_tree,search.grid[x,"max_depth"],search.grid[x,"eta"],search.grid[x,"lambda"],search.grid[x,"alpha"],fit_cv[min_tree,]))
}

tune_parameters_mlogloss <- do.call(rbind.data.frame,lapply(1:nrow(search.grid),get_cv_mlogloss))
tune_parameters_mlogloss[,index:=1]
tune_parameters_mlogloss[,index2:=cumsum(index),by=test.mlogloss.mean]
tune_parameters_mlogloss[,index3:=1]
tune_parameters_mlogloss[index2>1,index3:=0]
tune_parameters_mlogloss[,index4:=cumsum(index3)]

print(search.grid[tune_parameters_mlogloss[order(test.mlogloss.mean),index4],],topn=100)
tuned_parameters <- tune_parameters_mlogloss[order(test.mlogloss.mean)]
save(tuned_parameters,file="secondTuning.rda")
