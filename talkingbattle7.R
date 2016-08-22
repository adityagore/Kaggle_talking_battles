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
#createColors <- grDevices::colorRampPalette(RColorBrewer::brewer.pal(9,"Set1"),interpolate="spline")


events.info <- fread("events.csv",colClasses = c("character","character","character","numeric","numeric"))
events.info[,timestamp:=parse_datetime(timestamp,"%Y-%m-%d %H:%M:%S")]
events.info[,weekhour:=(wday(timestamp)-1)*24+hour(timestamp)+(minute(timestamp)/60)+(second(timestamp)/3600)]
events.info[,':='(weekday=wday(timestamp,label = TRUE),timeday=hour(timestamp))]
events.info[,timestamp:=NULL]
events.info[,partofday:= as.character(cut(timeday,breaks=c(0,5,12,17,21,24),labels = c("Night","Morning","Afternoon","Evening","Night"),right = FALSE))]

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
# China.long <- map_data("world","China")[,"long"]
# China.lat <- map_data("world","China")[,"lat"]
# # SDMTools::pnt.in.poly(events.info[device_id=="4505543221133337925",.(longitude,latitude)],map_data("world","China")[,c("long","lat")])
# events.info[,inChina:=point.in.polygon(longitude,latitude,China.long,China.lat)]
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

# devices.with.app.info <- events.apps.info[,.(length=.N,NAs=sum(is.na(app_id))),by=.(device_id)][length>1 & NAs>0,device_id]
# events.apps.info <- events.apps.info[!(device_id%in%devices.with.app.info&is.na(app_id))]
gc()

# devices.with.noapp.info <- events.apps.info[is.na(app_id),device_id]
devices.with.noapp.info <- events.apps.info[,.(length=.N,NAs=sum(is.na(app_id))),by=.(device_id)][length==NAs][,device_id]

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
events.apps.info[,missingEvents:=ifelse(is.na(event_id),1,0)]

devices.with.noapp.info <- events.apps.info[missingEvents==0,.(length=.N,NAs=sum(is.na(app_id))),by=.(device_id)][length==NAs][,device_id]
events.apps.info[,missingApps:=ifelse(device_id%in%devices.with.noapp.info,1,0)]
events.apps.info[,':='(region=NA,regionLabel="NA")]

world.data <- map_data("world")

for(REGION in unique(world.data$region)){
  start.time <- Sys.time()
  print(which(unique(world.data$region)==REGION,arr.ind = TRUE))
  print(REGION)
  temp.region <- world.data %>% filter(region==REGION)
  events.apps.info[(!is.na(longitude))&(regionLabel=="NA"),region:=point.in.polygon(longitude,latitude,temp.region$long,temp.region$lat)]
  expr <- parse(text=paste0("regionLabel:=\"",REGION,"\""))
  events.apps.info[(!is.na(longitude))&(region==1)&(regionLabel=="NA"),eval(expr)]
  print(paste0("Done ",REGION))
  gc()
  end.time <- Sys.time()
  print(paste0("Started: ",start.time))
  print(paste0("Total time taken: ", end.time-start.time))
}


# world.data <- rworldmap::getMap(resolution = "high")@data %>% select(continent,REGION,LON,LAT) %>% filter(!is.na(continent))
# 
# for(region in unique(world.data$REGION)){
#   temp.region <- world.data %>% filter(REGION==region)
#   events.apps.info[(!is.na(longitude))&regionLabel=="NA",region:=point.in.polygon(longitude,latitude,temp.region$LON,temp.region$LAT)]
#   expr <- parse(text=paste0("regionLabel:=\"",region,"\""))
#   events.apps.info[(!is.na(longitude))&(region==1)&(regionLabel=="NA"),eval(expr)]
#   gc()
# }
# 
# for(Continent in unique(world.data$continent)){
#   temp.continent <- world.data %>% filter(continent==Continent)
#   events.apps.info[(!is.na(longitude))&continentLabel=="NA",continent:=point.in.polygon(longitude,latitude,temp.continent$LON,temp.continent$LAT)]
#   expr <- parse(text=paste0("continentLabel:=\"",Continent,"\""))
#   events.apps.info[(!is.na(longitude))&(continent==1)&(continentLabel=="NA"),eval(expr)]
#   gc()
# }


# events.apps.info[events.apps.info[, sample(.I[!is.na(latitude)], 10000)],]%>% ggplot()+borders()+theme_bw()+geom_point(aes(x=longitude,y=latitude,color=regionLabel))



events.apps.summary <- events.apps.info[,.(Nactivity=sum(is_active,na.rm = TRUE),Pactivity=mean(is_active,na.rm=TRUE),missingEvents=max(missingEvents),missingApps=max(missingApps)),by=.(device_id,weekday,partofday,app_id,categories,timeday)]
setkey(events.apps.summary,device_id)
rm(events.apps.info)
gc()
events.apps.summary[Nactivity==0,Pactivity:=0]
final.merge <- events.apps.summary[,':='(phone_brand=total.info[device_id,phone_brand],device_model=total.info[device_id,device_model],group=total.info[device_id,group],gender=total.info[device_id,gender],age=total.info[device_id,age])]
setkey(final.merge,device_id)
rm(events.apps.summary)
gc()


final.merge[,week_day:=paste0(as.character(weekday),":",as.character(partofday))]

device.categories <- final.merge[!is.na(app_id),.(N=sum(Nactivity),P=mean(Pactivity)),by=.(device_id,app_id,categories)]
device.categories[grepl("unknown,unknown",categories),categories:=gsub("unknown,unknown","unknown",categories)]
device.categories[,length:=length(strsplit(unique(categories),split=",")[[1]]),by=app_id]
device.categories[,app.group:=cut(length,breaks = c(0,3,9,30),labels = c("Small","Average","Big"))]

device.categories.length <- device.categories[,.(freq=sum(N)),by=.(device_id,app.group)]


categories <- strsplit(device.categories[,categories],",")
device.categories.new <- data.table("device_id"=rep(device.categories[,device_id],unlist(lapply(categories,length))),"N"=rep(device.categories[,N],unlist(lapply(categories,length))),"P"=rep(device.categories[,P],unlist(lapply(categories,length))),"categories"=unlist(categories))

device.categories <- device.categories.new[,.(N=sum(N),P=mean(P)),by=.(device_id,categories)]
# device.categories <- unique(device.categories)
# device.categories[,Pcategories:=Ncategories/sum(Ncategories),by=device_id]

device.weekday.activities <- final.merge[!is.na(app_id),.(Nactivity=sum(Nactivity,na.rm=TRUE),Pactivity=mean(Pactivity,na.rm=TRUE)),by=.(device_id,weekday)]
device.partofday.activities <- final.merge[!is.na(app_id),.(Nactivity=sum(Nactivity,na.rm=TRUE),Pactivity=mean(Pactivity,na.rm=TRUE)),by=.(device_id,partofday)]
device.week_day.activities <- final.merge[!is.na(app_id),.(Nactivity=sum(Nactivity,na.rm=TRUE),Pactivity=mean(Pactivity,na.rm=TRUE)),by=.(device_id,week_day)]


device.events <- final.merge[,.(missingEvents=max(missingEvents)),by=device_id]

device.app.activities <- final.merge[!is.na(app_id),.(Nactivity=sum(Nactivity,na.rm=TRUE),Pactivity=mean(Pactivity,na.rm=TRUE)),by=.(device_id,app_id)]
device.missing.apps <- final.merge[,.(missingApps=max(missingApps,na.rm=TRUE)),by=device_id]




unique_devices <- total.info[,device_id] %>% unique()
# unique_devices_with_events <- final.merge[missingEvents==0,device_id] %>% unique()
# unique_mapps <- final.merge[missingEvents==0,app_id] %>% unique()
unique_apps <- final.merge[!is.na(app_id),app_id] %>% unique()
unique_brands <- final.merge[,phone_brand] %>% unique()
unique_models <- final.merge[,device_model] %>% unique()
unique_weekday <- final.merge[missingEvents==0,weekday]%>%unique()
unique_day <- final.merge[missingEvents==0,partofday]%>%unique()
unique_week_day <- final.merge[missingEvents==0,week_day] %>% unique()
unique_categories <- device.categories[,categories]%>% unique()
unique_app_group <- device.categories.length[,app.group] %>% unique()


gc()



id_devices <- match(total.info[,device_id],unique_devices)
id_brands <- match(total.info[,phone_brand],unique_brands)
id_models <- match(total.info[,device_model],unique_models) 

X_brand_full <- sparseMatrix(i=id_devices,j=id_brands,x=1,dimnames = list(unique_devices,unique_brands))
X_models_full <- sparseMatrix(i=id_devices,j=id_models,x=1,dimnames = list(unique_devices,unique_models))

id_devices <- match(device.app.activities[,device_id],unique_devices)
id_apps <- match(device.app.activities[,app_id],unique_apps)

X_apps_full <- sparseMatrix(i=id_devices,j=id_apps,x=1,dimnames = list(unique_devices,unique_apps),dims = c(length(unique_devices),length(unique_apps)))
X_apps_activities_full.log <- sparseMatrix(i=id_devices,j=id_apps,x=log1p(device.app.activities[,Nactivity]),dimnames = list(unique_devices,unique_apps),dims = c(length(unique_devices),length(unique_apps)))
X_apps_activities_full <- sparseMatrix(i=id_devices,j=id_apps,x=(device.app.activities[,Nactivity]),dimnames = list(unique_devices,unique_apps),dims = c(length(unique_devices),length(unique_apps)))
X_apps_activities_full.P <- sparseMatrix(i=id_devices,j=id_apps,x=(device.app.activities[,Pactivity]),dimnames = list(unique_devices,unique_apps),dims = c(length(unique_devices),length(unique_apps)))
X_apps_activities_full.Pm <- sparseMatrix(i=id_devices,j=id_apps,x=expm1(device.app.activities[,Pactivity]),dimnames = list(unique_devices,unique_apps),dims = c(length(unique_devices),length(unique_apps)))


id_devices <- match(device.categories[,device_id],unique_devices)
id_categories <- match(device.categories[,categories],unique_categories)

X_categories_full <- sparseMatrix(i=id_devices,j=id_categories,x=1,dimnames = list(unique_devices,unique_categories),dims = c(length(unique_devices),length(unique_categories)))
X_categories_activities_full <- sparseMatrix(i=id_devices,j=id_categories,x=device.categories[,N],dimnames = list(unique_devices,unique_categories),dims = c(length(unique_devices),length(unique_categories)))
X_categories_activities_full.log <- sparseMatrix(i=id_devices,j=id_categories,x=log1p(device.categories[,N]),dimnames = list(unique_devices,unique_categories),dims = c(length(unique_devices),length(unique_categories)))
X_categories_activities_full.P <- sparseMatrix(i=id_devices,j=id_categories,x=device.categories[,P],dimnames = list(unique_devices,unique_categories),dims = c(length(unique_devices),length(unique_categories)))
X_categories_activities_full.Pm <- sparseMatrix(i=id_devices,j=id_categories,x=expm1(device.categories[,P]),dimnames = list(unique_devices,unique_categories),dims = c(length(unique_devices),length(unique_categories)))


id_weekdays <- match(device.weekday.activities[,weekday],unique_weekday)
id_devices <- match(device.weekday.activities[,device_id],unique_devices)


X_weekdays_full <- sparseMatrix(i=id_devices,j=id_weekdays,x=1,dimnames = list(unique_devices,unique_weekday),dims = c(length(unique_devices),length(unique_weekday)))
X_weekdays_activities_full <- sparseMatrix(i=id_devices,j=id_weekdays,x=device.weekday.activities[,Nactivity],dimnames = list(unique_devices,unique_weekday),dims = c(length(unique_devices),length(unique_weekday)))
X_weekdays_activities_full.log <- sparseMatrix(i=id_devices,j=id_weekdays,x=log1p(device.weekday.activities[,Nactivity]),dimnames = list(unique_devices,unique_weekday),dims = c(length(unique_devices),length(unique_weekday)))
X_weekdays_activities_full.P <- sparseMatrix(i=id_devices,j=id_weekdays,x=device.weekday.activities[,Pactivity],dimnames = list(unique_devices,unique_weekday),dims = c(length(unique_devices),length(unique_weekday)))
X_weekdays_activities_full.Pm <- sparseMatrix(i=id_devices,j=id_weekdays,x=expm1(device.weekday.activities[,Pactivity]),dimnames = list(unique_devices,unique_weekday),dims = c(length(unique_devices),length(unique_weekday)))

id_partofdays <- match(device.partofday.activities[,partofday],unique_day)
id_devices <- match(device.partofday.activities[,device_id],unique_devices)

X_partofdays_full <- sparseMatrix(i=id_devices,j=id_partofdays,x=1,dimnames = list(unique_devices,unique_day),dims = c(length(unique_devices),length(unique_day)))
X_partofdays_activities_full <- sparseMatrix(i=id_devices,j=id_partofdays,x=device.partofday.activities[,Nactivity],dimnames = list(unique_devices,unique_day),dims = c(length(unique_devices),length(unique_day)))
X_partofdays_activities_full.log <- sparseMatrix(i=id_devices,j=id_partofdays,x=log1p(device.partofday.activities[,Nactivity]),dimnames = list(unique_devices,unique_day),dims = c(length(unique_devices),length(unique_day)))
X_partofdays_activities_full.P <- sparseMatrix(i=id_devices,j=id_partofdays,x=device.partofday.activities[,Pactivity],dimnames = list(unique_devices,unique_day),dims = c(length(unique_devices),length(unique_day)))
X_partofdays_activities_full.Pm <- sparseMatrix(i=id_devices,j=id_partofdays,x=expm1(device.partofday.activities[,Pactivity]),dimnames = list(unique_devices,unique_day),dims = c(length(unique_devices),length(unique_day)))




id_week_days <- match(device.week_day.activities[,week_day],unique_week_day)
id_devices <- match(device.week_day.activities[,device_id],unique_devices)

X_week_days_full <- sparseMatrix(i=id_devices,j=id_week_days,x=1,dimnames = list(unique_devices,unique_week_day),dims = c(length(unique_devices),length(unique_week_day)))
X_week_days_activities_full <- sparseMatrix(i=id_devices,j=id_week_days,x=device.week_day.activities[,Nactivity],dimnames = list(unique_devices,unique_week_day),dims = c(length(unique_devices),length(unique_week_day)))
X_week_days_activities_full.log <- sparseMatrix(i=id_devices,j=id_week_days,x=log1p(device.week_day.activities[,Nactivity]),dimnames = list(unique_devices,unique_week_day),dims = c(length(unique_devices),length(unique_week_day)))
X_week_days_activities_full.P <- sparseMatrix(i=id_devices,j=id_week_days,x=device.week_day.activities[,Pactivity],dimnames = list(unique_devices,unique_week_day),dims = c(length(unique_devices),length(unique_week_day)))
X_week_days_activities_full.Pm <- sparseMatrix(i=id_devices,j=id_week_days,x=expm1(device.week_day.activities[,Pactivity]),dimnames = list(unique_devices,unique_week_day),dims = c(length(unique_devices),length(unique_week_day)))

id_devices <- match(device.events[,device_id],unique_devices)


X_missingEvents <- sparseMatrix(i=id_devices,j=rep(1,length(id_devices)),x=device.events[,missingEvents],dim=c(length(unique_devices),1),dimnames = list(unique_devices,"missingEvents"))

id_devices <- match(device.missing.apps[,device_id],unique_devices)

X_missingApps <- sparseMatrix(i=id_devices,j=rep(1,length(id_devices)),x=device.missing.apps[,missingApps],dim=c(length(unique_devices),1),dimnames = list(unique_devices,"missingApps"))


prev_na_action <- options("na.action")
options(na.action = "na.pass")

matrixNames <- c("X_brand_full"=TRUE,"X_models_full"=TRUE,"X_apps_full"=TRUE,"X_apps_activities_full"=FALSE,"X_apps_activities_full.log"=FALSE,"X_apps_activities_full.P"=FALSE, "X_apps_activities_full.Pm"=FALSE,"X_categories_full"=TRUE,"X_categories_activities_full"=FALSE,"X_categories_activities_full.log"=FALSE,"X_categories_activities_full.P"=FALSE,"X_categories_activities_full.Pm"=FALSE,"X_partofdays_full"=TRUE,"X_partofdays_activities_full"=FALSE,"X_partofdays_activities_full.log"=FALSE,"X_partofdays_activities_full.P"=FALSE,"X_partofdays_activities_full.Pm"=FALSE,"X_week_days_full"=TRUE,"X_week_days_activities_full"=FALSE,"X_week_days_activities_full.log"=TRUE,"X_week_days_activities_full.P"=FALSE,"X_week_days_activities_full.Pm"=TRUE,"X_weekdays_full"=TRUE,"X_weekdays_activities_full"=FALSE, "X_weekdays_activities_full.log"=FALSE,"X_weekdays_activities_full.P"=FALSE,"X_weekdays_activities_full.Pm"=FALSE,"X_missingEvents"=TRUE,"X_missingApps"=TRUE)
expr <- parse(text = paste0("cbind(",paste0(names(matrixNames[matrixNames]),collapse = ","),")"))
# 
X_full <- eval(expr)
# X_full <-cbind(device_brand.matrix,device_models.matrix,device_app.matrix,device_categories.matrix,device_categories_size.matrix)
Y_full <- total.info[rownames(X_full),group]
Y_full <- factor(Y_full)

X_train <- X_full[!is.na(Y_full),]
X_test <- X_full[is.na(Y_full),]

Y_train <- Y_full[!is.na(Y_full)]
Y_test <- Y_full[is.na(Y_full)]
# 

ftrain <- xgb.DMatrix(X_train,label=as.integer(Y_train)-1,missing=NA)

# fit.glmnet <- glmnet(x=X_train,y=Y_train,family="multinomial",alpha=0.5,intercept = FALSE)
# fit.glmnet.cv <- cv.glmnet(X_train,Y_train,alpha=0.5,nfolds = 5,family="multinomial",intercept=FALSE)


depth <- 6
shrk <- 0.025
ntree <- 1000


param <- list(booster="gblinear",
              num_class=length(unique(Y_train)),
              objective="multi:softprob",
              eval_metric="mlogloss",
              eta=shrk,
              max.depth=depth,
              lambda=1,
              alpha=2,
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


ntree <- 110
fit.xgb <- xgb.train(params=param,
                     data=ftrain,
                     nrounds=ntree,
                     watchlist=watchlist,
                     verbose=1)

ftest <- xgb.DMatrix(X_test,label=as.integer(Y_test),missing=NA)


fit.glmnet <- cv.glmnet(X_train,Y_train,nfolds = 5,parallel = TRUE,family="multinomial",alpha=1)


pred <- predict(fit.xgb,ftest)
pred <- t(matrix(pred,nrow=length(levels(Y_full))))
colnames(pred) <- levels(Y_full)
first_submit <- cbind(device_id=rownames(X_test),as.data.frame(pred))
write.csv(first_submit,file="submit1.1.csv",row.names=F,quote=F)

pred <- predict(fit.glmnet,X_test,s="lambda.min",type="response")
pred <- t(matrix(pred,nrow=length(levels(Y_full))))
colnames(pred) <- levels(Y_full)
first_submit <-data.frame(device_id=rownames(pred[,,1]),as.data.frame(pred[,,1]))
colnames(first_submit) <- c("device_id",levels(Y_full))
write.csv(first_submit,file="submit1.2.csv",row.names=F,quote=F)


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


# fit_cv <- xgb.cv(params=param, data=ftrain, nrounds=100000, watchlist=watchlist, nfold=5, early.stop.round=3, verbose=1)



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

#missing events didn't help
