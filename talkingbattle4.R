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

# mygoogleapikey <- "AIzaSyBAFhy9_3dgJHmnDPFhDD2BQiKMDx8c0Hg"
# mapurl <- "https://maps.googleapis.com/maps/api/geocode/json?"
# 
# #####################################
# ### Trial###################
# #####################################
# latlng <- c(35.19,118.16)
# newurl <- paste0(mapurl,"latlng=",paste0(latlng,collapse = ","),"&sensor=true","&key=",mygoogleapikey)
# connection <- url(newurl)
# data <- RJSONIO::fromJSON(paste(readLines(connection), collapse=""))
# close(connection)
# data.json <- unlist(data)
# if(data.json["status"]=="OK"){
#   data.address <- data.json["results.formatted_address"]
#   data.address <- strsplit(data.address,",")[[1]]
#   data.address <- stringr::str_trim(data.address)
#   data.country <- data.address[length(data.address)]
#   data.region <- data.address[length(data.address)-1]
# }
# ###############################

events.info <- fread("events.csv",colClasses = c("character","character","character","numeric","numeric"))
events.info[,timestamp:=parse_datetime(timestamp,"%Y-%m-%d %H:%M:%S")]
events.info[,weekhour:=(wday(timestamp)-1)*24+hour(timestamp)+(minute(timestamp)/60)+(second(timestamp)/3600)]
events.info[,':='(weekday=wday(timestamp,label = TRUE),timeday=hour(timestamp))]
events.info[,timestamp:=NULL]

# ###########################################
# ######## Lat Long experiment  ###############
# ########################################
# 
# mygoogleapikey <- "AIzaSyBAFhy9_3dgJHmnDPFhDD2BQiKMDx8c0Hg"
# mapurl <- "https://maps.googleapis.com/maps/api/geocode/json?"
# getRegionCountry <- function(lat,long){
#   makeurl <- paste0(mapurl,"latlng=",lat,",",long,"&sensor=true","&key=",mygoogleapikey)
#   connection <- url(makeurl)
#   data <- fromJSON(paste(readLines(connection),collapse = ""))
#   close(connection)
#   data.json <- unlist(data)
#   if(data.json["status"]=="OK"){
#     data.address <- data.json["results.formatted_address"]
#     data.address <- strsplit(data.address,",")[[1]]
#     data.address <- str_trim(data.address)
#     add.length <- length(data.address)
#     if(add.length>4){
#       data.country <- data.address[length(data.address)-1]
#       data.region <- data.address[length(data.address)-2]
#     } else {
#     data.country <- data.address[length(data.address)]
#     data.region <- data.address[length(data.address)-1]
#     }
#     return(paste0(data.region,",",data.country))
#   } else{
#     return("NA")
#   }
# }
# 
# latlongs <- unique(events.info[,.(latitude,longitude)])
# latlongs[,regionCountry:=getRegionCountry(latitude,longitude),by=1:nrow(latlongs)]
# save(latlongs,file="latlongs.rda")
# #############################################################################

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
# events.apps.info[,':='(timestamp=NULL,longitude=NULL,latitude=NULL)]
rm(app.events.info,events.info)
gc()

devices.with.app.info <- events.apps.info[,.(length=.N,NAs=sum(is.na(app_id))),by=.(device_id,app_id)][length>1 & NAs>0,device_id]
# events.apps.info <- events.apps.info[!(device_id%in%devices.with.app.info&is.na(app_id))]
gc()

devices.with.noapp.info <- events.apps.info[is.na(app_id),device_id]
devices.with.noapp.info <- setdiff(devices.with.noapp.info,devices.with.app.info)

# events.apps.inChina <- events.apps.info[inChina>0]
# events.apps.outChina <- events.apps.info[inChina==0]
# rm(events.apps.info)
# gc()
# events.apps.summary.inChina <- events.apps.inChina[,.(Nactivity=sum(is_active,na.rm=TRUE),Pactivity=mean(is_active,na.rm=TRUE)),by=.(device_id,app_id,categories)]
# events.apps.summary.outChina <- events.apps.outChina[,.(Nactivity=sum(is_active,na.rm=TRUE),Pactivity=mean(is_active,na.rm=TRUE)),by=.(device_id,app_id,categories)]
# rm(devices.with.app.info,events.apps.inChina,events.apps.outChina)
# gc()

# final.merge.inChina <- merge(events.apps.inChina,total.info,by="device_id",all.y=TRUE)
# setkeyv(events.apps.inChina,c("device_id","event_id"))
# setkeyv(events.apps.outChina,c("device_id","event_id"))
# gc()
# events.apps.inChina[,':='(gender=total.info[device_id,gender],age=total.info[device_id,age],group=total.info[device_id,group],phone_brand=total.info[device_id,phone_brand],device_model=total.info[device_id,device_model])]
# setkey(events.apps.info,device_id)

# final.merge <- merge(events.apps.info,total.info,by="device_id",all.y=TRUE)

# rm(events.apps.inChina)
# gc()
# save(final.merge.inChina,file="BigMerge1.rda")
# rm(final.merge.inChina)
# gc()
# final.merge.outChina <- merge(events.apps.outChina,total.info,by="device_id",all.y=TRUE)
# setkeyv(final.merge.inChina,c("device_id","app_id"))
# setkeyv(final.merge.outChina,c("device_id","app_id"))
# rm(events.apps.summary.inChina,events.apps.summary.outChina)
# gc()
# final.merge.inChina[!is.na(app_id),app_id:=paste0("App ID (China):",app_id)]
# final.merge.outChina[!is.na(app_id),app_id:=paste0("App ID:",app_id)]
# final.merge.inChina[,':='(phone_brand=paste0("Brand (China):",phone_brand),device_model=paste0("Model:",device_model))]
# final.merge.outChina[,':='(phone_brand=paste0("Brand:",phone_brand),device_model=paste0("Model:",device_model))]
# 
# device.categories <- events.apps.info[!is.na(app_id),.(categories=paste0(unique(categories),collapse=",")),by=device_id]
# categories <- strsplit(device.categories[,categories],",")
# 
# device.categories <- data.table("device_id"=rep(device.categories[,device_id],unlist(lapply(categories,length))),"categories"=unlist(categories))
# 
# device.apps <- events.apps.info[,.(app_id=paste0(unique(app_id),collapse=",")),by=device_id]
# app_ids <- strsplit(device.apps[,app_id],",")
# device.apps <- data.table("device_id"=rep(device.apps[,device_id],unlist(lapply(app_ids,length))),"app_id"=unlist(app_ids))
# 





# id_devices <- match(final.merge.inChina[,device_id],unique_devices)
# id_devices.outChina <- match(final.merge.outChina[,device_id],unique_devices)
# id_mapps.inChina <- match(final.merge.inChina[,app_id],unique_mapps)
# id_mapps.outChina <- match(final.merge.outChina[,app_id],unique_mapps)
# id_apps.inChina <- match(final.merge.inChina[,app_id],unique_apps)
# id_apps.outChina <- match(final.merge.outChina[,app_id],unique_apps)
# id_brands.inChina <- match(final.merge.inChina[,phone_brand],unique_brands)
# id_brands.outChina <- match(final.merge.outChina[,phone_brand],unique_brands)
# id_models.inChina <- match(final.merge.inChina[,device_model],unique_models)
# id_models.outChina <- match(final.merge.outChina[,device_model],unique_models)
# id_categories.inChina <- match(device.categories.inChina[,categories],unique_categories)
# id_categories.outChina <- match(device.categories.outChina[,categories],unique_categories)
# 
# id_devices_apps.inChina <- cbind(id_devices.inChina,id_apps.inChina) %>% unique()
# id_devices_apps.inChina <- id_devices_apps.inChina[complete.cases(id_devices_apps.inChina),]
# id_devices_mapps.inChina <- cbind(id_devices.inChina,id_mapps.inChina) %>% unique()
# id_devices_brands.inChina <- cbind(id_devices.inChina,id_brands.inChina) %>% unique()
# id_devices_models.inChina <- cbind(id_devices.inChina,id_models.inChina) %>% unique()



# 
# 
# id_devices_apps.outChina <- cbind(id_devices.outChina,id_apps.outChina) %>% unique()
# id_devices_apps.outChina <- id_devices_apps.outChina[complete.cases(id_devices_apps.outChina),]
# id_devices_mapps.outChina <- cbind(id_devices.outChina,id_mapps.outChina) %>% unique()
# id_devices_brands.outChina <- cbind(id_devices.outChina,id_brands.outChina) %>% unique()
# id_devices_models.outChina <- cbind(id_devices.outChina,id_models.outChina) %>% unique()
# 
# id_devices.c.inChina <- match(device.categories.inChina[,device_id],unique_devices)
# id_devices.c.outChina <- match(device.categories.outChina[,device_id],unique_devices)
# 
# id_devices_categories.inChina <- cbind(id_devices.c.inChina,id_categories.inChina)
# id_devices_categories.outChina <- cbind(id_devices.c.outChina,id_categories.outChina)

setkey(events.apps.info,device_id)
events.apps.info <- events.apps.info[total.info[,device_id]]
setkey(events.apps.info,device_id)
gc()
events.apps.info[,':='(missingEvents=ifelse(is.na(event_id),1,0),missingApps=ifelse(device_id%in%devices.with.noapp.info,1,0),missingActivity=ifelse((device_id%in%devices.with.app.info)&is.na(app_id),1,0),missingTime=ifelse((device_id%in%devices.with.app.info)&is.na(app_id),weekhour,0))]
gc()
# for(i in seq_along(events.apps.info)){
#   set(events.apps.info,i=which(is.na(events.apps.info[,i,with=FALSE])),j=i,value = 0)
# }

for(name in names(total.info)){
  # if(!grepl("device_id",name)){
  if(name %in% c("phone_brand","device_model","group")){
    expr <- parse(text = paste0(name,":=total.info[device_id,\"",name,"\",with=FALSE]"))
    events.apps.info[,eval(expr)]
  }
  gc()
}


for(i in seq_along(events.apps.info)){
  if(class(events.apps.info[[i]])=="integer"){
    print("Changing Integer")
    set(events.apps.info,i=which(is.na(events.apps.info[[i]])),j=i,value = 0L)
  } else if(class(events.apps.info[[i]])=="numeric"){
    print("Changing Numeric")
    set(events.apps.info,i=which(is.na(events.apps.info[[i]])),j=i,value = 0.00)
  } else if(class(events.apps.info[[i]])=="character"){
    print("Character")
  }
}
events.apps.info[,':='(event_id=NULL,longitude=NULL,latitude=NULL,categories=NULL)]
events.apps.info[,weekhour:=is_active*weekhour]
setkey(events.apps.info,"device_id")
size <- 2^16

gc()

bigFormula <- formula(~inChina+app_id+is_active+weekhour+missingEvents+missingApps+missingActivity+missingTime+phone_brand+device_model-1)
X_test <- hashed.model.matrix(bigFormula,data=events.apps.info[is.na(group)],size)
gc()
X_train <- hashed.model.matrix(bigFormula,data=events.apps.info[!is.na(group)],size)
gc()

Y_train <- factor(events.apps.info[!is.na(group),group])
Y_test <- factor(events.apps.info[is.na(group),group])
# 
# unique_devices <- total.info[,device_id]%>%unique()
# unique_mapps <- events.apps.info[,app_id] %>% unique()
# # unique_apps <- c(final.merge.inChina[!is.na(app_id),app_id],final.merge.outChina[!is.na(app_id),app_id]) %>% unique()
# unique_brands <- total.info[,phone_brand]%>% unique()
# unique_models <- total.info[,device_model] %>% unique()
# # unique_categories <- c(device.categories.inChina[,categories],device.categories.outChina[,categories]) %>% unique()
# 
# id_devices_events <- match(events.apps.info[,device_id],unique_devices)
# id_apps_events <- match(events.apps.info[,app_id],unique_mapps)
# 
# id_devices_events_nna <- match(events.apps.info[!is.na(app_id),device_id],unique_devices)
# id_apps_events_nna <- match(events.apps.info[!is.na(app_id),app_id],unique_mapps[-1])
# 
# id_devices_events_na <- match(events.apps.info[is.na(app_id)&(!is.na(weekhour)),device_id],unique_devices)
# # id_apps_events_na <- match(events.apps.info[is.na(app_id),app_id],unique_mapps[-1])
# 
# device_app.matrix <- sparseMatrix(i=1:events.apps.info[,.N],j=id_apps_events,x=rep(1,nrow(events.apps.info)),dimnames = list(unique_devices[id_devices_events],unique_mapps))
# 
# activetime <- events.apps.info[!is.na(app_id),weekhour*is_active]
# 
# nna_rows <- events.apps.info[!is.na(app_id),.N]
# na_rows <- events.apps.info[is.na(app_id)&(!is.na(weekhour)),.N]
# device_app_activetime <- sparseMatrix(i=1:nna_rows,j=id_apps_events_nna,x=activetime,dimnames = list(unique_devices[id_devices_events_nna],paste("Active Time:",unique_mapps[-1])))
# device_app_inactivetime <- sparseVector(i=1:na_rows,j=id_apps_events_na,x=rep(1,na_rows),dimnames = list(unique_devices[id_devices_events_nna],paste("Inactive Time:",unique_mapps[-1])))
# 
# 
# id_brands <- match(total.info[events.apps.info[,device_id],phone_brand],unique_brands)
# 
# device_brands.matrix <- sparseMatrix(i=1:nrow(events.apps.info),j=id_brands,x=rep(1,nrow(events.apps.info)),dimnames = list(unique_devices[id_devices_events],unique_brands))
# 
# id_models <- match(total.info[events.apps.info[,device_id],device_model],unique_models)
# 
# device_models.matrix <- sparseMatrix(i=1:nrow(events.apps.info),j=id_models,x=rep(1,nrow(events.apps.info)),dimnames = list(unique_devices[id_devices_events],unique_models))
# 
# # id_devices_apps_events <- cbind(id_devices_events=1:nrow(events.apps.info),id_apps_events)
# # unique_mapps[is.na(unique_mapps)] <- "Missing Apps"
# # device_app.matrix <- Matrix(0,nrow=nrow(events.apps.info),ncol=length(unique_mapps),dimnames = list(unique_devices[id_devices_events],unique_mapps),sparse = TRUE)
# # device_app.matrix[id_devices_apps_events] <- 1
# 
# 
# 
# 
# 
# device_app.matrix <- Matrix(0,nrow=length(unique_devices),ncol=length(unique_mapps),dimnames=list(unique_devices,unique_mapps),sparse = TRUE)
# device_app.matrix[id_devices_mapps.inChina] <- 1
# device_app.matrix[id_devices_mapps.outChina] <- 1
# device_app_activity.matrix <- Matrix(0,nrow=length(unique_devices),ncol=length(unique_apps),dimnames=list(unique_devices,unique_apps),sparse = TRUE)
# 
# eps <- 1e-8
# device_app_activity.matrix[id_devices_apps.inChina] <- (final.merge.inChina[!is.na(app_id),Pactivity])
# device_app_activity.matrix[id_devices_apps.outChina] <- (final.merge.outChina[!is.na(app_id),Pactivity])
# 
# device_brand.matrix <- Matrix(0,nrow=length(unique_devices),ncol=length(unique_brands),dimnames=list(unique_devices,unique_brands),sparse = TRUE)
# device_brand.matrix[id_devices_brands.inChina] <- 1
# 
# device_models.matrix <- Matrix(0,nrow=length(unique_devices),ncol=length(unique_models),dimnames=list(unique_devices,unique_models),sparse = TRUE)
# device_models.matrix[id_devices_models.inChina] <- 1
# device_models.matrix[id_devices_models.outChina] <- 1
# 
# device_categories.matrix <- Matrix(0,nrow=length(unique_devices),ncol=length(unique_categories),dimnames=list(unique_devices,unique_categories),sparse = TRUE) 
# device_categories.matrix[id_devices_categories.inChina] <- 1
# device_categories.matrix[id_devices_categories.outChina] <- 1
# 
# device_categories_size.matrix <- Matrix(0,nrow=length(unique_devices),ncol=length(unique_categories),dimnames=list(unique_devices,unique_categories),sparse = TRUE)
# device_categories_size.matrix[id_devices_categories.inChina] <- log(device.categories.inChina[,Ncategories]+eps)
# device_categories_size.matrix[id_devices_categories.outChina] <- log(device.categories.outChina[,Ncategories]+eps)
# 
# # X_full <- cbind(device_brand.matrix,device_models.matrix,device_app.matrix,device_app_activity.matrix)
# X_full <-cbind(device_brand.matrix,device_models.matrix,device_app.matrix,device_categories.matrix,device_categories_size.matrix)
# Y_full <- total.info[rownames(X_full),group]
# Y_full <- factor(Y_full)
# 
# X_train <- X_full[!is.na(Y_full),]
# X_test <- X_full[is.na(Y_full),]
# 
# Y_train <- Y_full[!is.na(Y_full)]
# Y_test <- Y_full[is.na(Y_full)]
# 

ftrain <- xgb.DMatrix(X_train,label=as.integer(Y_train)-1,missing=NA)


depth <- 6
shrk <- 0.03
ntree <- 500


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
              nthread=2)


watchlist <- list(train=ftrain)

set.seed(1812)

fit_cv <- xgb.cv(params=param,
                 data=ftrain,
                 nrounds=ntree,
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
pred1 <- t(matrix(pred,nrow=length(levels(Y_train))))
colnames(pred1) <- levels(Y_train)
first_submit <- as.data.table(pred1)
first_submit[,device_id:=device_ids]
second_submit <- first_submit[,lapply(.SD,mean),by=device_id,.SDcols=names(first_submit)[-13]]
write.csv(second_submit,file="submit0.6.csv",row.names=F,quote=F)

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
