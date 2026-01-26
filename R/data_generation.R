
#' @export
shapes.func = function(X){
  tf <- get("tf", envir = asNamespace("SPQRX"))
  1/(1+exp(-(1-5*X[,1]*X[,2])))
}

#' @export
scales.func = function(X){
  tf <- get("tf", envir = asNamespace("SPQRX"))
  5*(1-1/(1+exp(-(1-5*X[,1]*X[,2]))))
}




#' @export
generate.simulation.dataset <- function(n = 1000, n.knots = 25)
{

  tf <- get("tf", envir = asNamespace("SPQRX"))

  x <- runif(3*n, 0, 1)
  x <- matrix(x,nrow = n,ncol = 3)


  shapes= shapes.func(x)

  scales= scales.func(x)

  y <- apply(cbind(scales,shapes),1,function(x) rlnorm(1,x[1],x[2]))



  n_validation <- n
  x_validation <- runif(3*n_validation, 0, 1)
  x_validation <- matrix(x_validation,nrow = n_validation, ncol = 3)

  shapes_validation= shapes.func(x_validation)
  scales_validation= scales.func(x_validation)
  y_validation <- apply(cbind(scales_validation,shapes_validation),1,function(x) rlnorm(1,x[1],x[2]))




  min.Y=0
  max.Y=max(y, y_validation)

  y <- (y-min.Y)/(max.Y-min.Y)
  y <- pmin(pmax(y, 1e-8), 1 - 1e-8)


  knots = quantile(y,probs=seq(1/(n.knots-2), 1-1/(n.knots-2), length=n.knots-3))
  #knots = seq(1/(n.knots-2), 1-1/(n.knots-2), length=n.knots-3)
  m_basis_training <- t(basis(y,n.knots,knots))
  i_basis_training <- t(basis(y,n.knots,knots, integral = TRUE))

  n.seq = 1001
  #y.seq<-c(0,exp(seq(log(1e-10), log(1), length.out = n.seq)))

  y.seq <- seq(0,1,length=n.seq)

  F.basis.seq <- tf$constant(basis(y.seq , n.knots,knots,integral = TRUE),dtype = 'float32') #this is used later to get quantiles
  f.basis.seq <- tf$constant(basis(y.seq , n.knots,knots),dtype = 'float32')
  y.seq <- tf$constant(y.seq, dtype = 'float32')
  y <- matrix(y,nrow = n)




  y_validation <- (y_validation-min.Y)/(max.Y-min.Y)
  m_basis_validation <- t(basis(y_validation,n.knots,knots))
  i_basis_validation <- t(basis(y_validation,n.knots,knots, integral = TRUE))

  y_validation <- matrix(y_validation,nrow = n_validation)

  ############ test data

  set.seed(1)
  n_test <- 5000
  x_testing <- runif(3*n_test, 0, 1)
  x_testing <- matrix(x_testing,nrow = n_test, ncol = 3)

  #Y_test <- rnorm(n_test, X_test, 0.8)
  shapes_test= shapes.func(x_testing)
  scales_test= scales.func(x_testing)

  y_testing <-  apply(cbind(scales_test,shapes_test),1,function(x) rlnorm(1,x[1],x[2]))



  y_testing <- (y_testing-min.Y)/(max.Y-min.Y)
  m_basis_testing <- t(basis(y_testing,n.knots,knots))
  i_basis_testing <- t(basis(y_testing,n.knots, knots, integral = TRUE))

  y_testing <- matrix(y_testing,nrow = n_test)



  #y.seq <- seq(0,1,length=n.seq)

  #F.basis.seq <- tf$constant(basis(y.seq , n.knots,knots,integral = TRUE),dtype = 'float32') #this is used later to get quantiles
  #f.basis.seq <- tf$constant(basis(y.seq , n.knots,knots),dtype = 'float32')
  #y.seq <- tf$constant(y.seq, dtype = 'float32')


  return (list(x_training = x, y_training = y, m_basis_training = m_basis_training, i_basis_training = i_basis_training,
               x_validation = x_validation, y_validation = y_validation, m_basis_validation = m_basis_validation,
               i_basis_validation = i_basis_validation, x_testing = x_testing, y_testing = y_testing, m_basis_testing = m_basis_testing,
               i_basis_testing = i_basis_testing, knots = knots, F.basis.seq = F.basis.seq, f.basis.seq = f.basis.seq, y.seq = y.seq))



}
