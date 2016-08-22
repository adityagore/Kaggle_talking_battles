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
China.long <- map_data("world","China")[,"long"]
China.lat <- map_data("world","China")[,"lat"]
# SDMTools::pnt.in.poly(events.info[device_id=="4505543221133337925",.(longitude,latitude)],map_data("world","China")[,c("long","lat")])
events.info[,inChina:=point.in.polygon(longitude,latitude,China.long,China.lat)]
app.events.info <- fread("app_events.csv",colClasses = c("character","character","integer","integer"))
setkeyv(app.events.info,c("event_id","app_id"))
app.labels.info <- fread("app_labels.csv",colClasses = c("character","character"))
app.labels.info <- unique(app.labels.info)
setkeyv(app.labels.info,c("app_id","label_id"))
label.categories.info <- fread("label_categories.csv",colClasses = c("character","character"))
setkey(label.categories.info,"label_id")
app.labels.categories <- merge(app.labels.info,label.categories.info,by="label_id",all.x=TRUE,allow.cartesian=TRUE)
app.labels.categories[,label_id:=NULL]
app.labels.categories <- app.labels.categories[,.(categories=paste0(category,collapse = ",")),by=app_id]
setkey(app.labels.categories,"app_id")
rm(app.labels.info,label.categories.info)
gc()
app.events.info[,categories:=app.labels.categories[app_id,categories]]
events.apps.info <- merge(events.info,app.events.info,by="event_id",all.x=TRUE,allow.cartesian = TRUE)
events.apps.info[,':='(timestamp=NULL,longitude=NULL,latitude=NULL)]
rm(app.events.info,events.info)
gc()

devices.with.app.info <- events.apps.info[,.(length=.N,NAs=sum(is.na(app_id))),by=.(device_id,app_id)][length>1 & NAs>0,device_id]
events.apps.info <- events.apps.info[!(device_id%in%devices.with.app.info&is.na(app_id))]
gc()

devices.with.noapp.info <- events.apps.info[is.na(app_id),device_id]

# events.apps.inChina <- events.apps.info[inChina>0]
# events.apps.outChina <- events.apps.info[inChina==0]
# rm(events.apps.info)
# gc()
# events.apps.summary.inChina <- events.apps.inChina[,.(Nactivity=sum(is_active,na.rm=TRUE),Pactivity=mean(is_active,na.rm=TRUE)),by=.(device_id,app_id,categories)]
# events.apps.summary.outChina <- events.apps.outChina[,.(Nactivity=sum(is_active,na.rm=TRUE),Pactivity=mean(is_active,na.rm=TRUE)),by=.(device_id,app_id,categories)]
# rm(devices.with.app.info,events.apps.inChina,events.apps.outChina)
# gc()

final.merge.inChina <- merge(events.apps.summary.inChina,total.info,by="device_id",all.y=TRUE)
final.merge.outChina <- merge(events.apps.summary.outChina,total.info,by="device_id",all.y=TRUE)
setkeyv(final.merge.inChina,c("device_id","app_id"))
setkeyv(final.merge.outChina,c("device_id","app_id"))
rm(events.apps.summary.inChina,events.apps.summary.outChina)
gc()
final.merge.inChina[!is.na(app_id),app_id:=paste0("App ID (China):",app_id)]
final.merge.outChina[!is.na(app_id),app_id:=paste0("App ID:",app_id)]
final.merge.inChina[,':='(phone_brand=paste0("Brand (China):",phone_brand),device_model=paste0("Model:",device_model))]
final.merge.outChina[,':='(phone_brand=paste0("Brand:",phone_brand),device_model=paste0("Model:",device_model))]

device.categories.inChina <- final.merge.inChina[!is.na(app_id),.(categories=paste0(categories,collapse=",")),by=device_id]
device.categories.outChina <- final.merge.outChina[!is.na(app_id),.(categories=paste0(categories,collapse=",")),by=device_id]
categories.inChina <- strsplit(device.categories.inChina[,categories],",")
categories.outChina <- strsplit(device.categories.outChina[,categories],",")

device.categories.inChina <- data.table("device_id"=rep(device.categories.inChina[,device_id],unlist(lapply(categories.inChina,length))),"categories"=unlist(categories.inChina))
device.categories.outChina <- data.table("device_id"=rep(device.categories.outChina[,device_id],unlist(lapply(categories.outChina,length))),"categories"=unlist(categories.outChina))

device.categories.inChina[,Ncategories:=.N,by=.(device_id,categories)]
device.categories.inChina <- unique(device.categories.inChina)
device.categories.inChina[,Pcategories:=Ncategories/sum(Ncategories),by=device_id]

device.categories.outChina[,Ncategories:=.N,by=.(device_id,categories)]
device.categories.outChina <- unique(device.categories.outChina)
device.categories.outChina[,Pcategories:=Ncategories/sum(Ncategories),by=device_id]



unique_devices <- c(final.merge.inChina[,device_id],final.merge.outChina[,device_id]) %>% unique()
unique_mapps <- c(final.merge.inChina[,app_id],final.merge.outChina[,app_id]) %>% unique()
unique_apps <- c(final.merge.inChina[!is.na(app_id),app_id],final.merge.outChina[!is.na(app_id),app_id]) %>% unique()
unique_brands <- c(final.merge.inChina[,phone_brand],final.merge.outChina[,phone_brand]) %>% unique()
unique_models <- c(final.merge.inChina[,device_model],final.merge.outChina[,device_model]) %>% unique()
unique_categories <- c(device.categories.inChina[,categories],device.categories.outChina[,categories]) %>% unique()

id_devices.inChina <- match(final.merge.inChina[,device_id],unique_devices)
id_devices.outChina <- match(final.merge.outChina[,device_id],unique_devices)
id_mapps.inChina <- match(final.merge.inChina[,app_id],unique_mapps)
id_mapps.outChina <- match(final.merge.outChina[,app_id],unique_mapps)
id_apps.inChina <- match(final.merge.inChina[,app_id],unique_apps)
id_apps.outChina <- match(final.merge.outChina[,app_id],unique_apps)
id_brands.inChina <- match(final.merge.inChina[,phone_brand],unique_brands)
id_brands.outChina <- match(final.merge.outChina[,phone_brand],unique_brands)
id_models.inChina <- match(final.merge.inChina[,device_model],unique_models)
id_models.outChina <- match(final.merge.outChina[,device_model],unique_models)
id_categories.inChina <- match(device.categories.inChina[,categories],unique_categories)
id_categories.outChina <- match(device.categories.outChina[,categories],unique_categories)

id_devices_apps.inChina <- cbind(id_devices.inChina,id_apps.inChina) %>% unique()
id_devices_apps.inChina <- id_devices_apps.inChina[complete.cases(id_devices_apps.inChina),]
id_devices_mapps.inChina <- cbind(id_devices.inChina,id_mapps.inChina) %>% unique()
id_devices_brands.inChina <- cbind(id_devices.inChina,id_brands.inChina) %>% unique()
id_devices_models.inChina <- cbind(id_devices.inChina,id_models.inChina) %>% unique()



id_devices_apps.outChina <- cbind(id_devices.outChina,id_apps.outChina) %>% unique()
id_devices_apps.outChina <- id_devices_apps.outChina[complete.cases(id_devices_apps.outChina),]
id_devices_mapps.outChina <- cbind(id_devices.outChina,id_mapps.outChina) %>% unique()
id_devices_brands.outChina <- cbind(id_devices.outChina,id_brands.outChina) %>% unique()
id_devices_models.outChina <- cbind(id_devices.outChina,id_models.outChina) %>% unique()

id_devices.c.inChina <- match(device.categories.inChina[,device_id],unique_devices)
id_devices.c.outChina <- match(device.categories.outChina[,device_id],unique_devices)

id_devices_categories.inChina <- cbind(id_devices.c.inChina,id_categories.inChina)
id_devices_categories.outChina <- cbind(id_devices.c.outChina,id_categories.outChina)

unique_mapps[is.na(unique_mapps)] <- "Missing Apps"

device_app.matrix <- Matrix(0,nrow=length(unique_devices),ncol=length(unique_mapps),dimnames=list(unique_devices,unique_mapps),sparse = TRUE)
device_app.matrix[id_devices_mapps.inChina] <- 1
device_app.matrix[id_devices_mapps.outChina] <- 1
device_app_activity.matrix <- Matrix(0,nrow=length(unique_devices),ncol=length(unique_apps),dimnames=list(unique_devices,unique_apps),sparse = TRUE)

eps <- 1e-8
device_app_activity.matrix[id_devices_apps.inChina] <- (final.merge.inChina[!is.na(app_id),Pactivity])
device_app_activity.matrix[id_devices_apps.outChina] <- (final.merge.outChina[!is.na(app_id),Pactivity])

device_brand.matrix <- Matrix(0,nrow=length(unique_devices),ncol=length(unique_brands),dimnames=list(unique_devices,unique_brands),sparse = TRUE)
device_brand.matrix[id_devices_brands.inChina] <- 1

device_models.matrix <- Matrix(0,nrow=length(unique_devices),ncol=length(unique_models),dimnames=list(unique_devices,unique_models),sparse = TRUE)
device_models.matrix[id_devices_models.inChina] <- 1
device_models.matrix[id_devices_models.outChina] <- 1

device_categories.matrix <- Matrix(0,nrow=length(unique_devices),ncol=length(unique_categories),dimnames=list(unique_devices,unique_categories),sparse = TRUE) 
device_categories.matrix[id_devices_categories.inChina] <- 1
device_categories.matrix[id_devices_categories.outChina] <- 1

device_categories_size.matrix <- Matrix(0,nrow=length(unique_devices),ncol=length(unique_categories),dimnames=list(unique_devices,unique_categories),sparse = TRUE)
device_categories_size.matrix[id_devices_categories.inChina] <- log(device.categories.inChina[,Ncategories]+eps)
device_categories_size.matrix[id_devices_categories.outChina] <- log(device.categories.outChina[,Ncategories]+eps)

# X_full <- cbind(device_brand.matrix,device_models.matrix,device_app.matrix,device_app_activity.matrix)
X_full <-cbind(device_brand.matrix,device_models.matrix,device_app.matrix,device_categories.matrix,device_categories_size.matrix)
Y_full <- total.info[rownames(X_full),group]
Y_full <- factor(Y_full)

X_train <- X_full[!is.na(Y_full),]
X_test <- X_full[is.na(Y_full),]

Y_train <- Y_full[!is.na(Y_full)]
Y_test <- Y_full[is.na(Y_full)]


ftrain <- xgb.DMatrix(X_train,label=as.integer(Y_train)-1,missing=NA)


depth <- 6
shrk <- 0.025
ntree <- 1000


param <- list(booster="gblinear",
              num_class=length(unique(Y_train)),
              objective="multi:softprob",
              eval_metric="mlogloss",
              eta=shrk,
              max.depth=depth,
              lambda=5,
              alpha=3,
              subsample=0.5,
              colsample_bytree=0.5,
              num_parallel_tree=2,
              nthread=3)


watchlist <- list(train=ftrain)

set.seed(1812)

fit_cv <- xgb.cv(params=param,
                 data=ftrain,
                 nrounds=1000,
                 watchlist=watchlist,
                 nfold=5,
                 early.stop.round=10,
                 verbose=1)
#

ntree <- 102
fit.xgb <- xgb.train(params=param,
                     data=ftrain,
                     nrounds=ntree,
                     watchlist=watchlist,
                     verbose=1)

ftest <- xgb.DMatrix(X_test,label=as.integer(Y_test),missing=NA)


pred <- predict(fit.xgb,ftest)
pred <- t(matrix(pred,nrow=length(levels(Y_full))))
colnames(pred) <- levels(Y_full)
first_submit <- cbind(device_id=rownames(X_test),as.data.frame(pred))
write.csv(first_submit,file="submit0.5.csv",row.names=F,quote=F)

search.grid <- expand.grid(
  max_depth = 6,
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
                nthread=1)
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
save(tuned_parameters,file="thirdTuning.rda")
