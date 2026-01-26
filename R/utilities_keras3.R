# SPQR functions
#' @export
basis <- function(y,K,knots, integral=FALSE){
  tf <- get("tf", envir = asNamespace("SPQRX"))
  library(splines2)
  B     <- mSpline(y, knots = knots,
                   Boundary.knots=c(0,1),
                   intercept=TRUE, degree = 2,
                   integral=integral)
  return(t(B))}
#' @export
spqrk <- function(pred,Y=NULL,nY=101,tau=0.5,type='QF'){
  # pred        <- as.matrix(model(X))
  tf <- get("tf", envir = asNamespace("SPQRX"))
  n <- nrow(pred)
  ntau = length(tau)
  n.knots <- ncol(pred)
  if(is.null(Y) | type=='QF')
    Y <- seq(0,1,length.out = nY)
  B <- (basis(Y , n.knots,knots, integral = (type!='PDF')))
  if(ncol(B)!=n)
    df1  <- pred%*%B
  if(ncol(B)==n)
    df1  <- colSums(B*t(pred))
  if(type!='QF'){
    return(df1)
  }
  if(type=='QF'){
    qf1 = matrix(NA,n,ntau)
    for(i in 1:n)
      qf1[i,] <- stats::approx(df1[i,], Y, xout=tau, ties = list("ordered", min))$y
    return(qf1)
  }
}
#' @export
F.SPQR=function(y){
  tf <- get("tf", envir = asNamespace("SPQRX"))
  y[y<0]=0
  y[y>1]=1
  spqrk(pred,Y=y,type = 'CDF')
}
#' @export
f.SPQR=function(y){
  tf <- get("tf", envir = asNamespace("SPQRX"))
  try(out<-spqrk(pred,Y=y,type = 'PDF') , silent=T)
  out[y<0]=0
  out[y>1]=0
  out
}
#' @export
Finv.SPQR=function(y){
  tf <- get("tf", envir = asNamespace("SPQRX"))
  spqrk(pred,tau =y,type = 'QF')


}

library(evd)
#' @export
F.GPD=function(y,u=0,scale=1,shape=0.1){
  tf <- get("tf", envir = asNamespace("SPQRX"))
  evd::pgpd(y,u,scale,shape)
}
#' @export
F.inverse.GPD=function(p,u=0,scale=1,shape=0.1){
  tf <- get("tf", envir = asNamespace("SPQRX"))
  u + scale * ((1-p)^(-shape) - 1)/shape
}
#' @export
f.GPD=function(y,u=0,scale=1,shape=0.1){
  tf <- get("tf", envir = asNamespace("SPQRX"))
  try(out<-(1/scale)*(1+shape*(y-u)/scale)^(-1/shape-1), silent=T)
  out[1+shape*(y-u)/scale<=0]=0
  out
}
#' @export
F.blend=function(y,F.S,F.G,p){
  tf <- get("tf", envir = asNamespace("SPQRX"))
  F.S^(1-p)*F.G^(p)
}
#' @export
weight=function(y,a,b,c1=5,c2=5){
  tf <- get("tf", envir = asNamespace("SPQRX"))
  pbeta((y-a)/(b-a),c1,c2)
}
#' @export
weight.prime=function(y,a,b,c1=5,c2=5){
  tf <- get("tf", envir = asNamespace("SPQRX"))
  dbeta((y-a)/(b-a),c1,c2)/(b-a)
}

#' @export
sigma.val=function(a,b,p.a,p.b,xi){
  tf <- get("tf", envir = asNamespace("SPQRX"))
  xi*(a-b)/(((1-p.a)^(-xi))-((1-p.b)^(-xi)))
}
#' @export
u.val=function(a,p.a,sigma,xi){
  tf <- get("tf", envir = asNamespace("SPQRX"))
  a-sigma/xi*((1-p.a)^(-xi)-1)
}

#' @export
f.blend=function(y,F.B,F.S,f.S,F.G,f.G,p,p.prime){
  tf <- get("tf", envir = asNamespace("SPQRX"))
  if(F.G==0) return(f.S)
  F.B*(p.prime*log(F.G)+p*f.G/F.G-p.prime*log(F.S)+(1-p)*f.S/F.S)
}

#' @export
f.blend.full=function(y){
  tf <- get("tf", envir = asNamespace("SPQRX"))
  dens=apply(as.matrix(y),1,function(x){
    F.S=F.SPQR(x)
    F.G=F.GPD(x,u=u,scale=sig,shape=xi)
    p=weight(x,a,b)
    F.B=F.blend(x,F.S,F.G,p)

    f.S=f.SPQR(x)
    f.G=f.GPD(x,u=u,scale=sig,shape=xi)
    p.prime=weight.prime(x,a,b)
    if(F.G==0) return(f.S)
    if(F.G>0) return(f.blend(x,F.B,F.S,f.S,F.G,f.G,p,p.prime))
  })
  return(dens)
}








#' @export
F_SPQR_inverse.tf=function(p,F_SPQR_seq, y_seq){
  tf <- get("tf", envir = asNamespace("SPQRX"))

  # K <- backend()

  d= tf$shape(F_SPQR_seq)[2]

  a_ind = tf$math$top_k(-abs(F_SPQR_seq-p), k=1L)$indices

  # a = y_seq[a_ind[,1  ]]
  a = tf$gather(y_seq, a_ind[,1])
  a.lower_ind = a_ind-1
  a.upper_ind = a_ind+1

  #Remove boundary indices
  a.lowest.bool = 1-tf$sign(a_ind)
  a.lower_ind = keras3::op_relu(a.lower_ind)
  a.highest.bool = 1-tf$sign(d- a.upper_ind)

  a.upper_ind = a.upper_ind -  a.upper_ind * a.highest.bool + a_ind * ( a.highest.bool)

  # a.lower = y_seq[a.lower_ind[,1  ]]
  a.lower = tf$gather(y_seq, a.lower_ind[,1])
  # a.upper = y_seq[a.upper_ind[,1  ]]
  a.upper = tf$gather(y_seq, a.upper_ind[,1])


  F.a = tf$gather_nd(batch_dims = 1, indices = a_ind,
                     params = F_SPQR_seq)
  F.a.lower = tf$gather_nd(batch_dims = 1, indices =a.lower_ind,
                           params = F_SPQR_seq)
  F.a.upper = tf$gather_nd(batch_dims = 1, indices = a.upper_ind,
                           params = F_SPQR_seq)

  m.upper = (F.a.upper - F.a)/(a.upper- a + tf$cast(a.highest.bool[,1], 'float32'))
  m.lower = (F.a.lower - F.a)/(a.lower- a + tf$cast(a.lowest.bool[,1], 'float32'))

  a = a + keras3::op_relu( p - F.a) / (m.upper + tf$cast(a.highest.bool[,1], 'float32')+(1-tf$sign( p - F.a) ))
  a = a + keras3::op_relu((F.a- p)) / (m.lower + tf$cast(a.lowest.bool[,1], 'float32') + (1-tf$sign(  F.a - p) ))

  return(a)
}




#' @export
sigma.val.tf=function(a,b,p.a,p.b,xi){
  tf <- get("tf", envir = asNamespace("SPQRX"))
  xi*(a-b)/(((1-p.a)^(-xi))-((1-p.b)^(-xi)))
}

#' @export
u.val.tf=function(a,p.a,sigma,xi){
  tf <- get("tf", envir = asNamespace("SPQRX"))
  a-sigma/xi*((1-p.a)^(-xi)-1)
}

#' @export
F.GPD.tf=function(y,u, scale, xi){
  tf <- get("tf", envir = asNamespace("SPQRX"))

  # K <- backend()
  y=keras3::op_relu(y-u)

  sigu=scale- scale*(1-tf$sign(y))+(1-tf$sign(y)) #If no exceedance, set sig to 1

  return( 1 - (1 + xi * y / sigu)^(-1/xi))

}

#' @export
f.GPD.tf=function(y,u, scale, xi){

  tf <- get("tf", envir = asNamespace("SPQRX"))
  # K <- backend()
  y=keras3::op_relu(y-u)

  sigu=scale- scale*(1-tf$sign(y))+(1-tf$sign(y)) #If no exceedance, set sig to 1

  #Evaluate log-likelihood
  ll1=-(1/xi+1)*tf$math$log1p(xi*y/sigu)

  #Uses non-zero response values only
  ll2= tf$math$log(sigu) *tf$sign(ll1)

  return(tf$math$exp(ll1+ll2)*(-tf$sign(ll1)))
}

#' @export
F.GPD.tf_seq=function(y.seq,u, scale, xi){

  tf <- get("tf", envir = asNamespace("SPQRX"))
  # K <- backend()
  y.seq = keras3::op_relu(t(y.seq)-t(t(u)))

  sigu=t(t(scale))- t(t(scale))*(1-tf$sign(y.seq))+(1-tf$sign(y.seq)) #If no exceedance, set sig to 1


  tmp = keras3::op_relu(1 + t(t(xi)) * y.seq / sigu)
  tmp = tmp^(-1/t(t(xi)))
  return( 1 - tmp)

}

#' @export
f.GPD.tf_seq=function(y.seq,u, scale, xi){

  tf <- get("tf", envir = asNamespace("SPQRX"))
  # K <- backend()

  y.seq = keras3::op_relu(t(y.seq)-t(t(u)))

  sigu=t(t(scale))- t(t(scale))*(1-tf$sign(y.seq))+(1-tf$sign(y.seq)) #If no exceedance, set sig to 1
  sigu=t(t(scale))- t(t(scale))*(1-tf$sign(y.seq))+(1-tf$sign(y.seq)) #If y less than endpoint, set to 1

  uF = keras3::op_relu(-scale/xi)
  xi.sign = keras3::op_relu(-tf$sign(xi))

  y.seq = y.seq -  y.seq*t(t(xi.sign))*keras3::op_relu(tf$sign(y.seq-t(t(uF))))
  #Evaluate log-likelihood
  ll1=-t(t(1/xi+1))*tf$math$log1p(t(t(xi))*y.seq/sigu)

  #Uses non-zero response values only
  ll2= tf$math$log(sigu) *tf$sign(ll1)
  return(tf$math$exp(ll1+ll2)*(-tf$sign(ll1)))
}

#' @export
weight.tf=function(y,a,b,c1=5,c2=5){

  tf <- get("tf", envir = asNamespace("SPQRX"))
  # K <- backend()


  temp=(y-a)/(b-a) #Need to set values <0 and >1 to 0 and 1, otherwise function breaks
  temp=keras3::op_relu(temp)
  temp=1-temp
  temp= keras3::op_relu(temp)
  temp=1-temp
  return(tf$math$betainc(c1,c2,temp))
}

#' @export
weight.prime.tf=function(y,a,b,c1=5,c2=5){

  tf <- get("tf", envir = asNamespace("SPQRX"))
  # K <- backend()

  temp=(y-a)/(b-a) #Need to set values <0 and >1 to 0 and 1, otherwise function breaks
  temp= keras3::op_relu(temp)
  temp=1-temp
  temp= keras3::op_relu(temp)
  temp=1-temp
  pprime = temp^(c1-1)*(1-temp)^(c2-1)/beta(c1,c2)
  pprime=pprime/(b-a)
  return( pprime)
}

#' @export
weight.tf_seq=function(y.seq,a,b,c1=5,c2=5){

  tf <- get("tf", envir = asNamespace("SPQRX"))
  # K <- backend()


  temp=(t(y.seq)-t(t(a)))/t(t(b-a)) #Need to set values <0 and >1 to 0 and 1, otherwise function breaks
  temp= keras3::op_relu(temp)
  temp=1-temp
  temp= keras3::op_relu(temp)
  temp=1-temp
  return(tf$math$betainc(c1,c2,temp))
}

#' @export
weight.prime.tf_seq=function(y.seq,a,b,c1=5,c2=5){
  tf <- get("tf", envir = asNamespace("SPQRX"))
  # K <- backend()

  temp=(t(y.seq)-t(t(a)))/t(t(b-a)) #Need to set values <0 and >1 to 0 and 1, otherwise function breaks
  temp= keras3::op_relu(temp)
  temp=1-temp
  temp= keras3::op_relu(temp)
  temp=1-temp
  pprime = temp^(c1-1)*(1-temp)^(c2-1)/beta(c1,c2)
  pprime=pprime/t(t(b-a))
  return( pprime)
}

#' @export
log.F.blend.tf=function(F.S,F.G,p){
  tf <- get("tf", envir = asNamespace("SPQRX"))
  # K <- backend()

  (1-p)*tf$math$log(F.S)+ p * tf$math$log(F.G)
}



#
# p_a=0.5
# p_b=0.975
#
# y_pred = model.heavy( list(X,y ,I_basis ))
# y_true = k_constant(M_basis)

# y_pred = model.heavy( list(X_validation,y_validation ,I_basis_validation ))
# y_true = k_constant(M_basis_validation)
#
# library(tensorflow)

# y_pred = model.heavy( list(X_test,y_test ,I_basis_test ))
# y_true = k_constant(M_basis_test)

#y.seq <- k_constant(seq(0,1,length=n.seq))

#' @export
nloglik_loss = function(F.basis.seq,f.basis.seq=NULL, y.seq,p_a=0.5,p_b=0.975,c1=5,c2=5, lambda= NULL){

  tf <- get("tf", envir = asNamespace("SPQRX"))
  if(is.null(lambda) | is.null(f.basis.seq)){
    loss = function (y_true, y_pred){
      # K <- backend()

      numbasis <- keras3::op_shape(y_true)[[2]]
      # print(numbasis)
      probs <- y_pred[,1:numbasis]

      xi <- y_pred[,numbasis+1]
      y_value <- y_pred[,numbasis+2]
      I_bases <- y_pred[,(numbasis+3):(2*numbasis+3)]
      M_bases <- y_true

      F_SPQR_seq = tf$matmul(probs, F.basis.seq)


      a=F_SPQR_inverse.tf(p_a,F_SPQR_seq , y_seq = y.seq)
      b=F_SPQR_inverse.tf(p_b,F_SPQR_seq , y_seq = y.seq)
      # a = tf$zeros(shape = K$shape(b))

      sigmas <-sigma.val.tf(a, b, p_a, p_b,xi)

      u<-u.val.tf(a,p_a,sigmas,xi)


      f.G = f.GPD.tf(y_value, u, sigmas, xi)
      F.G = F.GPD.tf(y_value, u, sigmas, xi)



      f.S = keras3::op_sum(M_bases*probs,axis=2)
      F.S = keras3::op_sum(I_bases*probs,axis=2)



      weights = weight.tf(y_value,a, b, c1, c2)


      weights.prime = weight.prime.tf(y_value,a, b, c1, c2)



      F.G_zero_ind = 1-tf$sign(F.G)

      F.G = F.G + 1* (F.G_zero_ind)

      F.S_zero_ind = 1-tf$sign(F.S)

      F.S = F.S + 1* (F.S_zero_ind)

      log.F.blend = log.F.blend.tf( F.S, F.G, weights)

      weights_one_ind = 1-tf$sign(1-weights)

      weights_zero_ind = 1-tf$sign(weights)

      weights.prime = weights.prime - weights.prime * weights_zero_ind
      weights.prime = weights.prime - weights.prime * weights_one_ind

      # Testing Block for making sure everything is good
      eps <- 1e-6

      F.G <- keras3::op_clip(F.G, eps, Inf)
      F.S <- keras3::op_clip(F.S, eps, Inf)
      f.G <- keras3::op_clip(f.G, eps, Inf)
      f.S <- keras3::op_clip(f.S, eps, Inf)
      weights <- keras3::op_clip(weights, eps, 1 - eps)
      # Testing Block


      log.f.blend =
        log.F.blend+
        tf$math$log(
          weights.prime*
            tf$math$log(F.G)+
            weights*f.G/F.G-
            weights.prime*tf$math$log(F.S)+
            (1-weights)*
            f.S/F.S
        )

      # This is
      # Testing Block
      #log_inner <-
      #  weights.prime * tf$math$log(F.G) +
      #  weights * f.G / F.G -
      #  weights.prime * tf$math$log(F.S) +
      #  (1 - weights) * f.S / F.S

      #log_inner <- keras3::op_clip(log_inner, eps, Inf)

      #log.f.blend <- log.F.blend + tf$math$log(log_inner)
      # Testing Block


      spqr_loss <- -keras3::op_average((log.f.blend))
      return(spqr_loss)

    }
  }else{
    loss = function (y_true, y_pred){
      # K <- backend()

      numbasis <- keras3::op_shape(y_true)[[2]]
      # print(numbasis)
      probs <- y_pred[,1:numbasis]

      xi <- y_pred[,numbasis+1]
      y_value <- y_pred[,numbasis+2]
      I_bases <- y_pred[,(numbasis+3):(2*numbasis+3)]
      M_bases <- y_true

      # print(op_dtype(probs))
      # print(op_dtype(F.basis.seq))

      F.S_seq = tf$matmul(probs, F.basis.seq)
      f.S_seq = tf$matmul(probs, f.basis.seq)


      a=F_SPQR_inverse.tf(p_a,F.S_seq , y_seq = y.seq)
      b=F_SPQR_inverse.tf(p_b,F.S_seq , y_seq = y.seq)
      # a = tf$zeros(shape = K$shape(b))

      sigmas <-sigma.val.tf(a, b, p_a, p_b,xi)

      u<-u.val.tf(a,p_a,sigmas,xi)


      f.G = f.GPD.tf(y_value, u, sigmas, xi)
      F.G = F.GPD.tf(y_value, u, sigmas, xi)


      f.G_seq = f.GPD.tf_seq(y.seq, u, sigmas, xi)
      F.G_seq = F.GPD.tf_seq(y.seq, u, sigmas, xi)

      f.S = keras3::op_sum(M_bases*probs,axis=2)
      F.S = keras3::op_sum(I_bases*probs,axis=2)


      weights = weight.tf(y_value,a, b, c1, c2)

      weights_seq=weight.tf_seq(y.seq,a, b, c1, c2)

      weights.prime = weight.prime.tf(y_value,a, b, c1, c2)

      weights.prime_seq = weight.prime.tf_seq(y.seq,a, b, c1, c2)


      F.G_zero_ind = 1-tf$sign(F.G)

      F.G = F.G + 1* (F.G_zero_ind)

      F.S_zero_ind = 1-tf$sign(F.S)

      F.S = F.S + 1* (F.S_zero_ind)

      log.F.blend = log.F.blend.tf( F.S, F.G, weights)

      weights_one_ind = 1-tf$sign(1-weights)

      weights_zero_ind = 1-tf$sign(weights)

      weights.prime = weights.prime - weights.prime * weights_zero_ind
      weights.prime = weights.prime - weights.prime * weights_one_ind


      # Testing Block
      eps <- 1e-6

      F.G <- keras3::op_clip(F.G, eps, Inf)
      F.S <- keras3::op_clip(F.S, eps, Inf)
      f.G <- keras3::op_clip(f.G, eps, Inf)
      f.S <- keras3::op_clip(f.S, eps, Inf)
      weights <- keras3::op_clip(weights, eps, 1 - eps)

      # Testing Block



      # Testing Block
      #log_inner <-
      #  weights.prime * tf$math$log(F.G) +
      #  weights * f.G / F.G -
      #  weights.prime * tf$math$log(F.S) +
      #  (1 - weights) * f.S / F.S

      #log_inner <- keras3::op_clip(log_inner, eps, Inf)

      #log.f.blend <- log.F.blend + tf$math$log(log_inner)
      # Testing Block


      log.f.blend =
        log.F.blend+
        tf$math$log(
          weights.prime*
            tf$math$log(F.G)+
            weights*f.G/F.G-
            weights.prime*tf$math$log(F.S)+
            (1-weights)*
            f.S/F.S
        )



      F.G_zero_ind_seq = 1-tf$sign(F.G_seq)

      F.G_seq = F.G_seq + 1* (F.G_zero_ind_seq)

      F.S_zero_ind_seq = 1-tf$sign(F.S_seq)

      F.S_seq = F.S_seq + 1* (F.S_zero_ind_seq)


      log.F.blend_seq = log.F.blend.tf( F.S_seq, F.G_seq, weights_seq)

      weights_one_ind_seq = 1-tf$sign(1-weights_seq)

      weights_zero_ind_seq = 1-tf$sign(weights_seq)

      weights.prime_seq = weights.prime_seq - weights.prime_seq * weights_zero_ind_seq
      weights.prime_seq = weights.prime_seq - weights.prime_seq * weights_one_ind_seq

      f.blend_seq =
        tf$math$exp(log.F.blend_seq)*
        (weights.prime_seq*
           tf$math$log(F.G_seq)+
           weights_seq*f.G_seq/F.G_seq-
           weights.prime_seq*tf$math$log(F.S_seq)+
           (1-weights_seq)*
           f.S_seq/F.S_seq)


      PENALTY = lambda* keras3::op_sum(op_relu(-f.blend_seq))


      spqr_loss <- -keras3::op_average((log.f.blend))
      return(spqr_loss+PENALTY)

    }

  }
  return(loss)

}

#' @export
nloglik_loss_SPQR  = function (y_true, y_pred){
  tf <- get("tf", envir = asNamespace("SPQRX"))
  # K <- backend()
  numbasis <- keras3::op_shape(y_true)[[2]]
  # print(numbasis)
  probs <- y_pred[,1:numbasis]
  xi <- y_pred[,numbasis+1]
  y_value <- y_pred[,numbasis+2]
  sumprod <- keras3::op_sum(y_true*probs,axis=2)
  spqr_loss <- -keras3::op_sum(keras3::op_log(sumprod))
  return(spqr_loss)
}

#' @export
predict.spqr <- function(model,covariates, knots, y=NULL,ny =1001,tau=0.5,type='QF'){
  tf <- get("tf", envir = asNamespace("SPQRX"))
  pred        <- as.matrix(model(list(covariates = covariates,
                                      data =matrix(0,nrow=nrow(covariates),ncol=1),
                                      I_basis = matrix(0, nrow = nrow(covariates), ncol = (length(knots) + 3)))))

  n <- nrow(pred)
  ntau = length(tau)
  n.knots <- (length(knots) + 3)
  probs <- matrix(pred[,1:n.knots],n,n.knots)

  if(is.null(Y) | type=='QF')
    y <- seq(0,1,length.out = nY)
  B <- (basis(y , n.knots,knots, integral = (type!='PDF')))
  if(ncol(B)!=n)
    df1  <- probs%*%B
  if(ncol(B)==n)
    df1  <- colSums(B*t(probs))
  if(type=='CDF' | type=="PDF"){
    df1[y<0]=0
    if(type =="PDF") df1[y>1]=0 else if(type=="CDF") df1[y>1] = 1
    return(df1)


  }
  if(type=='QF'){
    qf1 = matrix(NA,n,ntau)
    for(i in 1:n)
      qf1[i,] <- stats::approx(df1[i,], y, xout=tau, ties = list("ordered", min))$y
    return(qf1)
  }
}


#' @export
predict.spqrk <- function(model,covariates,I_basis, knots, Y=NULL,nY=1001,tau=0.5,type='QF'){
  tf <- get("tf", envir = asNamespace("SPQRX"))
  pred        <- as.matrix(model(list(covariates = covariates,
                                      data =matrix(0,nrow=nrow(covariates),ncol=1),
                                      I_basis = I_basis)))
  n <- nrow(pred)
  ntau = length(tau)
  n.knots <- dim(I_basis)[2]
  probs <- matrix(pred[,1:n.knots],n,n.knots)

  if(is.null(Y) | type=='QF')
    Y <- seq(0,1,length.out = nY)
  B <- (basis(Y , n.knots,knots, integral = (type!='PDF')))
  if(ncol(B)!=n)
    df1  <- probs%*%B
  if(ncol(B)==n)
    df1  <- colSums(B*t(probs))
  if(type=='CDF' | type=="PDF"){
    df1[Y<0]=0
    if(type =="PDF") df1[Y>1]=0 else if(type=="CDF") df1[Y>1] = 1
    return(df1)


  }
  if(type=='QF'){
    qf1 = matrix(NA,n,ntau)
    for(i in 1:n)
      qf1[i,] <- stats::approx(df1[i,], Y, xout=tau, ties = list("ordered", min))$y
    return(qf1)
  }
}

#' @export
predict.spqrk.GPD <- function(model,
                              covariates,
                              I_basis,
                              M_basis,
                              knots,
                              Y = NULL,
                              nY = 10000,
                              tau = 0.5,
                              type = 'QF',
                              c1 = 10,
                              c2 = 10,
                              p_a = 0.5,
                              p_b = 0.975) {
  tf <- get("tf", envir = asNamespace("SPQRX"))

  pred        <- as.matrix(model(list(
    covariates = covariates,
    data = matrix(0, nrow = nrow(covariates), ncol = 1),
    I_basis = I_basis
  )))
  n <- nrow(pred)
  ntau = length(tau)
  n.knots <- dim(I_basis)[2]
  probs <- matrix(pred[, 1:n.knots], n, n.knots)
  xi = pred[, n.knots + 1]

  if (is.null(Y)) {
    Y <- seq(0, 1, length.out = nY)
  }
  if (type == "QF") {
    quant = matrix(0, ncol = ntau, nrow = n)
  }
  if (type == "QF" & any(tau <= p_a)) {
    Y.Q.est <- seq(0, 1, length.out = nY)
    quant = predict.spqrk(
      model = model,
      type = 'QF',
      Y = Y.Q.est,
      knots = knots,
      covariates = covariates,
      I_basis = I_basis,
      tau = tau
    )
    quant[, tau > p_a] = NA
    if (sum(tau <= p_a) == ntau)
      return(quant)
  }


  ab = predict.spqrk(
    model = model,
    type = 'QF',
    Y = Y,
    knots = knots,
    covariates = covariates,
    I_basis = I_basis,
    tau = c(p_a, p_b)
  )
  if (type == "PDF" & length(ab) == 2) {
    ab = matrix(
      rep(c(ab), length = 2 * length(Y)),
      nrow = length(Y),
      ncol = 2,
      byrow = T
    )
  }

  sig <- c(sigma.val(ab[, 1], ab[, 2], p_a, p_b, xi))
  u <- c(u.val(ab[, 1], p_a, sig, xi))

  if (type != "QF") {
    I_basis <- basis(Y , n.knots, knots, integral = TRUE)
    M_basis <- basis(Y , n.knots, knots, integral = FALSE)
    if (ncol(I_basis) != n)
      F.S  <- probs %*% I_basis
    if (ncol(I_basis) == n)
      F.S  <- colSums(I_basis * t(probs))
    F.S[F.S > 1] = 1

    p = weight(Y, ab[, 1], ab[, 2], c1 = c1, c2 = c2)
    F.G = apply(cbind(Y, u, sig, xi), 1, function(x)
      F.GPD(x[1], x[2], x[3], x[4]))

    F.B =  F.blend(Y, F.S, F.G, p)
  }



  if (type == "QF" & any(tau >= p_b)) {
    quant[, tau >= p_b] = apply(as.matrix(tau[tau >= p_b]),1,
                                function(x) F.inverse.GPD(x, u, sig, xi))
    if (sum(tau > p_a & tau < p_b) == 0)
      return(quant)
  }



  if (type == "QF" & any(tau > p_a & tau < p_b)) {
    tau.sub = tau[tau > p_a & tau < p_b]
    for (i in 1:n) {
      Y <- seq(min(ab[i, 1]), max(ab[i, 2]), length.out = nY)

      I_basis <- basis(Y , n.knots, knots, integral = TRUE)
      M_basis <- basis(Y , n.knots, knots, integral = FALSE)
      if (ncol(I_basis) != n)
        F.S  <- probs[i, ] %*% I_basis
      if (ncol(I_basis) == n)
        F.S  <- colSums(I_basis * t(probs[i, ]))
      F.S[F.S > 1] = 1

      p = weight(Y, ab[i, 1], ab[i, 2], c1 = c1, c2 = c2)
      F.G = apply(cbind(Y, u[i], sig[i], xi[i]), 1, function(x)
        F.GPD(x[1], x[2], x[3], x[4]))

      F.B =  F.blend(Y, F.S, F.G, p)


      quant[i, tau > p_a &
              tau < p_b] = stats::approx(F.B, Y, xout = tau.sub, ties = list("ordered", min))$y
    }
    return(quant)
  }


  if (type == "CDF")
    return(F.B)
  if (type == "PDF") {
    if (ncol(M_basis) != n)
      f.S  <- probs %*% M_basis
    if (ncol(M_basis) == n)
      f.S  <- colSums(M_basis * t(probs))
    f.S[Y < 0] = 0
    f.S[Y > 1] = 0
    f.G = apply(cbind(Y, u, sig, xi), 1, function(x)
      f.GPD(x[1], x[2], x[3], x[4]))
    p.prime = weight.prime(Y, ab[, 1], ab[, 2], c1 = c1, c2 = c2)

    f.B = F.B * (p.prime * log(F.G) + p * f.G / F.G - p.prime * log(F.S) +
                   (1 - p) * f.S / F.S)
    f.B[F.G == 0] = f.S[F.G == 0]
    return (f.B)
  }



}

#' @export
predict.spqrk.GPD <- function(model,
                              covariates,
                              I_basis,
                              M_basis,
                              knots,
                              Y = NULL,
                              nY = 10000,
                              tau = 0.5,
                              type = 'QF',
                              c1 = 10,
                              c2 = 10,
                              p_a = 0.5,
                              p_b = 0.975) {
  tf <- get("tf", envir = asNamespace("SPQRX"))

  .pdf.function <- function(x, probs_row, M_basis_row) {
    # Approximate PDF using M_basis and probs
    m_basis <- basis(x, n.knots, knots, integral = FALSE)
    pdf_val <- probs_row %*% m_basis
    as.numeric(pdf_val)
  }


  .expectation.function <- function(x, probs_row) {
    I_basis <- basis(x, n.knots, knots, integral = TRUE)
    F_x <- probs_row %*% I_basis
    1 - F_x   # survival function
  }

  .second.moment.function <- function(x, probs_row) {
    I_basis <- basis(x, n.knots, knots, integral = TRUE)
    F_x <- probs_row %*% I_basis
    2 * x * (1 - F_x)
  }


  pred        <- as.matrix(model(list(
    covariates = covariates,
    data = matrix(0, nrow = nrow(covariates), ncol = 1),
    I_basis = I_basis
  )))
  n <- nrow(pred)
  ntau = length(tau)
  n.knots <- dim(I_basis)[2]
  probs <- matrix(pred[, 1:n.knots], n, n.knots)
  xi = pred[, n.knots + 1]

  if (is.null(Y)) {
    Y <- seq(0, 1, length.out = nY)
  }
  if (type == "QF") {
    quant = matrix(0, ncol = ntau, nrow = n)
  }
  if (type == "QF" & any(tau <= p_a)) {
    Y.Q.est <- seq(0, 1, length.out = nY)
    quant = predict.spqrk(
      model = model,
      type = 'QF',
      Y = Y.Q.est,
      knots = knots,
      covariates = covariates,
      I_basis = I_basis,
      tau = tau
    )
    quant[, tau > p_a] = NA
    if (sum(tau <= p_a) == ntau)
      return(quant)
  }


  ab = predict.spqrk(
    model = model,
    type = 'QF',
    Y = Y,
    knots = knots,
    covariates = covariates,
    I_basis = I_basis,
    tau = c(p_a, p_b)
  )
  if (type == "PDF" & length(ab) == 2) {
    ab = matrix(
      rep(c(ab), length = 2 * length(Y)),
      nrow = length(Y),
      ncol = 2,
      byrow = T
    )
  }

  sig <- c(sigma.val(ab[, 1], ab[, 2], p_a, p_b, xi))
  u <- c(u.val(ab[, 1], p_a, sig, xi))

  if (type != "QF") {
    I_basis <- basis(Y , n.knots, knots, integral = TRUE)
    M_basis <- basis(Y , n.knots, knots, integral = FALSE)
    if (ncol(I_basis) != n)
      F.S  <- probs %*% I_basis
    if (ncol(I_basis) == n)
      F.S  <- colSums(I_basis * t(probs))
    F.S[F.S > 1] = 1

    p = weight(Y, ab[, 1], ab[, 2], c1 = c1, c2 = c2)
    F.G = apply(cbind(Y, u, sig, xi), 1, function(x)
      F.GPD(x[1], x[2], x[3], x[4]))

    F.B =  F.blend(Y, F.S, F.G, p)
  }



  if (type == "QF" & any(tau >= p_b)) {
    quant[, tau >= p_b] = apply(as.matrix(tau[tau >= p_b]),1,
                                function(x) F.inverse.GPD(x, u, sig, xi))
    if (sum(tau > p_a & tau < p_b) == 0)
      return(quant)
  }



  if (type == "QF" & any(tau > p_a & tau < p_b)) {
    tau.sub = tau[tau > p_a & tau < p_b]
    for (i in 1:n) {
      Y <- seq(min(ab[i, 1]), max(ab[i, 2]), length.out = nY)

      I_basis <- basis(Y , n.knots, knots, integral = TRUE)
      M_basis <- basis(Y , n.knots, knots, integral = FALSE)
      if (ncol(I_basis) != n)
        F.S  <- probs[i, ] %*% I_basis
      if (ncol(I_basis) == n)
        F.S  <- colSums(I_basis * t(probs[i, ]))
      F.S[F.S > 1] = 1

      p = weight(Y, ab[i, 1], ab[i, 2], c1 = c1, c2 = c2)
      F.G = apply(cbind(Y, u[i], sig[i], xi[i]), 1, function(x)
        F.GPD(x[1], x[2], x[3], x[4]))

      F.B =  F.blend(Y, F.S, F.G, p)


      quant[i, tau > p_a &
              tau < p_b] = stats::approx(F.B, Y, xout = tau.sub, ties = list("ordered", min))$y
    }
    return(quant)
  }





  if (type == "CDF")
    return(F.B)
  if (type == "PDF") {
    if (ncol(M_basis) != n)
      f.S  <- probs %*% M_basis
    if (ncol(M_basis) == n)
      f.S  <- colSums(M_basis * t(probs))
    f.S[Y < 0] = 0
    f.S[Y > 1] = 0
    f.G = apply(cbind(Y, u, sig, xi), 1, function(x)
      f.GPD(x[1], x[2], x[3], x[4]))
    p.prime = weight.prime(Y, ab[, 1], ab[, 2], c1 = c1, c2 = c2)

    f.B = F.B * (p.prime * log(F.G) + p * f.G / F.G - p.prime * log(F.S) +
                   (1 - p) * f.S / F.S)
    f.B[F.G == 0] = f.S[F.G == 0]
    return (f.B)
  }



}








