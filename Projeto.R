#Cromossoma:
# ny nx N n1 n2 n3 n4 n5 function
library(keras)
library(ggplot2)
library(GA)
library(R.matlab)
library(gramEvol)


#Clean console
rm(list=ls(all=TRUE))
cat("\014")  
n1=n2=n3=n4=n5=1

#setwd("/home/oem/Downloads/ 2018_sysid/narmax-sysid2018")
#devtools::load_all()
#library(narmax)

use_session_with_seed(8)

nu = 2
ny = 2

#devtools::load_all()

# Alterar aqui abaixo com o path para ambos documents .R que estao na pasta
source("/home/oem/Downloads/oipe18-master - Copia/fun_sys_id.R")
source("/home/oem/Downloads/oipe18-master - Copia/plot_xcorrel.R")

# Alterar aqui com o path para os dados
auxu = readMat("/home/oem/Downloads/2018 AST - engine/data11.mat")

y = auxu$RPM[,1]
u = auxu$Fuel[,1]

ndata = length(y)

#Normalize input and output
y <- scale(y, center = mean(y), scale = sd(y))
u <- scale(u, center = mean(u), scale = sd(u))

# ratio for training and test set
trPct = 0.5
ntr = round(trPct*ndata)

#Devide data for training
ytr = y[1:ntr]
utr = u[1:ntr]
#ytr = y[1:20]
#utr = u[1:20]

#Devide data for training
yte = y[(ntr+1):ndata]
ute = u[(ntr+1):ndata]
#yte = y[20+1:40]
#ute = u[20+1:40]


#learning rate
lr = 0.001 


funcao <- function(ny,nu,n1,n2,n3,n4,n5,fa)
{
  ny = round(ny)
  nu = round(nu)
  n1 = round(n1)
  n2 = round(n2)
  n3 = round(n3)
  n4 = round(n4)
  n5 = round(n5)
  fa = round(fa)
  
  maxn = max(c(ny,nu));

  # create reg matrix
  list[all_data, all_targets]     = create_reg_matrix(y,u,ny,nu)
  list[train_data, train_targets] = create_reg_matrix(ytr,utr,ny,nu)
  #list[test_data, test_targets]   = create_reg_matrix(yte,ute,ny,nu)
  
  if(fa==0) 
  {
    funcao_ativacao <- "tanh"
  } 
  else if(fa==1)
  {
    funcao_ativacao <- "sigmoid"
  }
  else 
  {
    funcao_ativacao <- "relu"
  }
  
 #Fazer as camadas da rede neural
  
 j <- 0 # usado para adicionar a camada de entrada
 
 camada<-"keras_model_sequential()" 
 camada_entrada <-", input_shape = dim(train_data)[[2]]"
 
 #n1
  if(n1>0)
  {
    camada<-paste(c(camada,"%>%layer_dense(units = n1, activation = funcao_ativacao"),collapse=" ")
    if(j==0)
    {
      camada<-paste(c(camada,camada_entrada),collapse=" ")
      j<-1
    }
  camada<-paste(c(camada,")"),collapse=" ")
 }
 
 #n2
 if(n2>0)
 {
   camada<-paste(c(camada,"%>%layer_dense(units = n2, activation = funcao_ativacao"),collapse=" ")
   if(j==0)
   {
     camada<-paste(c(camada,camada_entrada),collapse=" ")
     j<-1
   }
   camada<-paste(c(camada,")"),collapse=" ")
 }
 
 #n3
 if(n3>0)
 {
   camada<-paste(c(camada,"%>%layer_dense(units = n3, activation = funcao_ativacao"),collapse=" ")
   if(j==0)
   {
     camada<-paste(c(camada,camada_entrada),collapse=" ")
     j<-1
   }
   camada<-paste(c(camada,")"),collapse=" ")
 }
 
 #n4
 if(n4>0)
 {
   camada<-paste(c(camada,"%>%layer_dense(units = n4, activation = funcao_ativacao"),collapse=" ")
   if(j==0)
   {
     camada<-paste(c(camada,camada_entrada),collapse=" ")
     j<-1
   }
   camada<-paste(c(camada,")"),collapse=" ")
 }
 
 #n5
 if(n5>0)
 {
   camada<-paste(c(camada,"%>%layer_dense(units = n5, activation = funcao_ativacao"),collapse=" ")
   if(j==0)
   {
     camada<-paste(c(camada,camada_entrada),collapse=" ")
     j<-1
   }
   camada<-paste(c(camada,")"),collapse=" ")
 }
 
 camada<-paste(c(camada,"%>% layer_dense(units = 1)"),collapse=" ")
 
 
 rede_neural <- parse(text=camada)
 model<-eval(rede_neural)
 
 model %>% compile(
  optimizer = optimizer_adam(lr=lr,amsgrad = TRUE,clipnorm=50), 
  loss = "mse", 
  metrics = c("mae")
 )
 
 model %>% fit(train_data, train_targets,
               epochs = 60, batch_size = 10,verbose = 1)
  
 yh       = predict_on_batch(model, x = all_data)
 #yh_train = predict_on_batch(model, x = train_data)
 #yh_test  = predict_on_batch(model, x = test_data)
 
 R2   = calc_R2(all_targets,yh[,]) 
 #R2tr = calc_R2(train_targets,yh_train[,])
 #R2te = calc_R2(test_targets,yh_test[,])
 
 # FREE RUN
 yh_fr       = predictFreeRun(u,y,ny,nu,model)
 #yh_train_fr = predictFreeRun(utr,ytr,ny,nu,model)
 #yh_test_fr  = predictFreeRun(ute,yte,ny,nu,model)
 
 R2_fr   = calc_R2(all_targets,yh_fr)
 #R2tr_fr = calc_R2(train_targets,yh_train_fr)
 #R2te_fr = calc_R2(test_targets,yh_test_fr)
 
 
 
 return(R2 + R2_fr)
}

#######



GA <- ga(type = "real-valued", 
         fitness =  function(x) funcao(x[1], x[2],x[3], x[4],x[5], x[6],x[7], x[8]),
         lower = c(1,1,0,0,0,0,0,0), upper = c(5,5,5,5,5,5,5,2),
         nBits = 3,
         popSize = 10, maxiter = 30, run = 30) #maxiter tava 1000 #popSize tava 10 #maxiter tava 100




#######################################################################################################


#Treinando a rede neural com os parâmetros otumos encontrados



#x <- GeneticAlg.int(genomeLen=2 ,codonMin = c(1,1,0,0,0,0,0,0), codonMax = c(5,5,5,5,5,5,5,2),
#                    allowrepeat = TRUE, evalFunc = funcao,terminationCost = 0.9)

#summary(GA)

#plot(GA)

# PLOT

x1=ny=round(4.6992)
x2=nu=round(2.390632)
x3=round(1.386573)
x4=round(1.652773)
x5=round(3.466085)
x6=round(3.25174)
x7=round(1.845383)
x8=round(0.4569116)


list[all_data, all_targets]     = create_reg_matrix(y,u,x1,x2)
list[train_data, train_targets] = create_reg_matrix(ytr,utr,x1,x2)
list[test_data, test_targets]   = create_reg_matrix(yte,ute,x1,x2)


build_model <- function(lr) {
  model <- keras_model_sequential() %>% 
    layer_dense(units = x3, activation = "tanh", input_shape = dim(train_data)[[2]]) %>% 
    layer_dense(units = x4, activation = "tanh") %>%
    layer_dense(units = x5, activation = "tanh") %>%
    layer_dense(units = x6, activation = "tanh") %>%
    layer_dense(units = x7, activation = "tanh") %>%
    layer_dense(units = 1) 
  
  model %>% compile(
    optimizer = optimizer_adam(lr=lr,amsgrad = TRUE,clipnorm=50), 
    loss = "mse", 
    metrics = c("mae")
  )
}

# Get a fresh, compiled model.
model <- build_model(lr = 0.001)

# Train it on the entirety of the data.
model %>% fit(train_data, train_targets,
              epochs = 100, batch_size = 16, verbose = 1)

maxn = max(c(x1,x2))
uxcorr = u[(maxn+1):ntr]  # u for corr tests.


# PREDICTIONS -------
# OSA
yh       = predict_on_batch(model, x = all_data)
yh_train = predict_on_batch(model, x = train_data)
yh_test  = predict_on_batch(model, x = test_data)

R2   = calc_R2(all_targets,yh[,]) 
R2tr = calc_R2(train_targets,yh_train[,])
R2te = calc_R2(test_targets,yh_test[,])

# FREE RUN
yh_fr       = predictFreeRun(u,y,ny,nu,model)
yh_train_fr = predictFreeRun(utr,ytr,ny,nu,model)
yh_test_fr  = predictFreeRun(ute,yte,ny,nu,model)

R2_fr   = calc_R2(all_targets,yh_fr)
R2tr_fr = calc_R2(train_targets,yh_train_fr)
R2te_fr = calc_R2(test_targets,yh_test_fr)

print(paste("R2",R2,"R2tr",R2tr,"R2te",R2te))
print(paste("R2_fr",R2_fr,"R2tr_fr",R2tr_fr,"R2te_fr",R2te_fr))


ndata = length(all_targets)

df = data.frame(
  t = rep(1:ndata,2),
  y = c(all_targets,yh_fr),
  Type = c(rep('Measured data',ndata),
           #rep('Free-run simulation',ndata),
           rep('Free Run Simulation',ndata))
)

ggplot(df,aes(x=t,y=y,color=Type)) + geom_line() +xlab('Sample')+ylab('Output (normalized)')

# grafico de Y vs Yh
df = data.frame(
  y = all_targets,
  #yh_fr = yh_fr,
  yh = yh[,]
)

p1 = ggplot(df,aes(x=y,y=yh)) + geom_point() +
  geom_abline(intercept = 0, slope = 1, color="blue", size=1.5)+
  xlab('Measured')+ylab('Predicted (one-step-ahead)')+ggtitle(paste("R2 =",R2))
p2 = ggplot(df,aes(x=y,y=yh_fr)) + geom_point() +
  geom_abline(intercept = 0, slope = 1, color="blue", size=1.5)+
  xlab('Measured')+ylab('Predicted (free-run)')+ggtitle(paste("R2 =",R2_fr))
multiplot(p1, p2, cols = 2)

###################################
# PLOT FR

df_fr = data.frame(
  x_fr=1:ndata, #Eixo x
  yh_fr = yh_fr, #Predict Free Run
  y=y[6:(ndata+5)] #Measured 
)

# PLOT OSA

df_OSA = data.frame(
  x_fr=1:ndata, #Eixo x
  yh = yh, #Predict OSA
  y=y[6:(ndata+5)] #Measured 
)

###################################

#Fre Run
ggplot(df_fr,aes(x_fr))+geom_line(aes(y=yh_fr,colour="Predicted"))+geom_line(aes(y=y,colour="Real"))+xlab("Time")+ylab("Output")+theme(text = element_text(size=25))+labs(color='')

#OSA
ggplot(df_OSA,aes(x_fr))+geom_line(aes(y=yh,colour="Predicted"))+geom_line(aes(y=y,colour="Real"))+xlab("Time")+ylab("Output")+theme(text = element_text(size=25))+labs(color='')


############################################
############################################
############################################
############################################
############################################
############################################
############################################

#use_session_with_seed(8)

#nrn = c(x3,x4,x5,x6,x7)
#acf = "tanh"
#mdl = ann(ny,nu,nrn,acf)

#mdl = estimate(mdl,ytr,utr,lr = 0.001, epochs = 100, batch_size = 16, verbose = 1)

#Pe1 = predict(mdl,y,u,K = 1) # one-step-ahead
#Pe0 = predict(mdl,y,u,K = 0) # free-run

#Pe1te = predict(mdl,yte,ute,K = 1) # one-step-ahead
#Pe0te = predict(mdl,yte,ute,K = 0) # free-run

#Pe1tr = predict(mdl,ytr,utr,K = 1) # one-step-ahead
#Pe0tr = predict(mdl,ytr,utr,K = 0) # free-run

#plot(Pe1$xcorrel)
#Pe0$plote
#Pe0$ploty
#Pe1$ploty

