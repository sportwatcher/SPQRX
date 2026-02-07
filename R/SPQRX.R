

#' @export
xi_custom_activation <- function(x) {
  0.5 * keras3::activation_sigmoid(x[keras3::all_dims(),1:1])#(0,0.5)
  # 0.5 * activation_tanh(x[all_dims(),1:1]) +0.1 #(-0.4,0.6)
}



#' @export
create.package.model <- function(model, n.knots, knots , p_a = NULL, p_b = NULL, c1 = NULL, c2 = NULL, normalizer = NULL,
                                 variable_names = NULL, spqrx = TRUE)
{
  return ( structure(
    list(model = model, n.knots = n.knots , knots = knots, p_a = p_a , p_b = p_b , c1 = c1 , c2 = c2, normalizer = normalizer, variable_names = variable_names, spqrx = spqrx),
    class = "spqrx_model"
  ))
}

#' @export
create.package.normalize.list <- function(x_std, x_mean, y_max, y_min)
{
  return (list(x_std = x_std, x_mean = x_mean, y_max = y_max, y_min = y_min))
}

#' @export
create.packages.hyperparameter <- function(p_a, p_b , c1 = 25, c2 = 5, batch_size = 500) {
  return (list(p_a = p_a, p_b = p_b, c1 = c1, c2 = c2, batch_size = batch_size))
}

#' @export
preprocessing.data <- function(x, y, n.knots,
                               testing_ratio = 0.1,
                               valid_ratio = 0.1,
                               normalize = FALSE)
{
  if (nrow(x) != nrow(y)) {
    stop("Dimensions of x and y do not match")
  }

  n <- nrow(x)
  testing_index <- sample(seq_len(n), floor(n * testing_ratio))

  x_training <- x[-testing_index, , drop = FALSE]
  x_testing  <- x[testing_index,  , drop = FALSE]

  y_training <- y[-testing_index, , drop = FALSE]
  y_testing  <- y[testing_index,  , drop = FALSE]

  normalizer <- NULL
  if(normalize){
    m.x <- apply(x_training, 2, mean)
    s.x <- apply(x_training, 2, sd)

    x_training <- scale(x_training, center = m.x, scale = s.x)
    x_testing  <- scale(x_testing,  center = m.x, scale = s.x)


    y_min <- min(y_training)
    y_max <- max(y_training)

    y_training <- (y_training - y_min) / (y_max - y_min)
    y_testing  <- (y_testing  - y_min) / (y_max - y_min)
    y <- (y - y_min) / (y_max - y_min)

    normalizer <- create.package.normalize.list(s.x, m.x, y_max, y_min)

  }



  n_train <- nrow(x_training)
  validation_index <- sample(seq_len(n_train),
                             floor(n_train * valid_ratio))

  x_validation <- x_training[validation_index, , drop = FALSE]
  y_validation <- y_training[validation_index, , drop = FALSE]

  x_training <- x_training[-validation_index, , drop = FALSE]
  y_training <- y_training[-validation_index, , drop = FALSE]


  probs <- seq(0, 1, length.out = n.knots)[-c(1, n.knots)]
  knots = quantile(y,probs=seq(1/(n.knots-2), 1-1/(n.knots-2), length=n.knots-3))


  if (normalize){
    return (
      list(
        x_training = x_training,
        x_testing = x_testing,
        x_validation = x_validation,
        y_training = y_training,
        y_testing = y_testing,
        y_validation = y_validation,
        m_basis_training = m_basis_training,
        m_basis_testing = m_basis_testing,
        m_basis_validation = m_basis_validation,
        i_basis_training = i_basis_training,
        i_basis_testing = i_basis_testing,
        i_basis_validation = i_basis_validation,
        knots = knots,
        normalizer = normalizer
      )
    )
  }else {

    return (
      list(
        x_training = x_training,
        x_testing = x_testing,
        x_validation = x_validation,
        y_training = y_training,
        y_testing = y_testing,
        y_validation = y_validation
      )
    )
  }

}

#' @export
SPQRX <- function(input_dim, hidden_dim, k) {

  tf <- get("tf", envir = asNamespace("SPQRX"))
  input_cov <- keras3::keras_input(shape = input_dim, name = "covariates")
  input_y   <- keras3::keras_input(shape = 1, name = "data")
  input_I   <- keras3::keras_input(shape = k, name = "I_basis")

  x <- input_cov
  for (h in hidden_dim) {
    x <- keras3::layer_dense(x, units = h, activation = "relu")
  }

  probs <- keras3::layer_dense(x, units = k, activation = "softmax", name = "probs")
  xi    <- keras3::layer_dense(x, units = 1, activation = xi_custom_activation, name = "xi")

  output <- keras3::op_concatenate(
    list(probs, xi, input_y, input_I),
    axis = -1
  ) |> keras3::layer_identity(name = "outs")

  keras3::keras_model(
    inputs  = list(input_cov, input_y, input_I),
    outputs = output,
    name    = "SPQR.heavy"
  )
}


#' @export
in.fit.spqr <- function(input_dim, hidden_dim, n.knots,knots, x_training, x_validation, y_training, y_validation,
                        m_basis_training, m_basis_validation, i_basis_training, i_basis_validation)
{


  tf <- get("tf", envir = asNamespace("SPQRX"))

  model <- SPQRX(input_dim, hidden_dim, n.knots)


  model |> keras3::compile(
    loss = nloglik_loss_SPQR,
    optimizer = keras3::optimizer_adam(learning_rate = 0.001)
  )

  c1 <- max(c1, 25)
  checkpoint <- keras3::callback_model_checkpoint(filepath=paste0('runs/','model','/spqr_initial.weights.h5'), monitor = "val_loss", verbose = 0,
                                                  save_best_only = TRUE, save_weights_only = TRUE, mode = "min",
                                                  save_freq = "epoch")




  history <- model |> keras3::fit(
    list(covariates = x_training, data = y_training, I_basis = i_basis_training),
    m_basis_training,
    epochs = 200,
    batch_size = 32,
    callbacks=list(checkpoint,keras3::callback_early_stopping(monitor = "val_loss",
                                                              min_delta = 0, patience = 5)),
    validation_data=list(
      list(covariates = x_validation, data = y_validation, I_basis = i_basis_validation),
      m_basis_validation)
  )


  #
  return (model)

}



#' @export
fit.spqr <- function(input_dim, hidden_dim, n.knots, x_training, x_validation, y_training, y_validation,pre_normalize = FALSE, package.it = TRUE)
{










  tf <- get("tf", envir = asNamespace("SPQRX"))
  y_max <- max(max(y_training), max(y_validation))
  y_min <- min(min(y_training), min(y_validation))

  y_training <- (y_training - y_min) /(y_max - y_min)
  y_validation <- (y_validation - y_min) / (y_max - y_min)


  x_combined <- rbind(x_training, x_validation)
  m.x <- apply(x_combined, 2, mean)
  s.x <- apply(x_combined, 2, sd)

  x_training <- scale(x_training, center = m.x, scale = s.x)
  x_validation <- scale(x_validation, center = m.x, scale = s.x)


  nas_sum <- sum ( is.na(x_training) )  + sum ( is.na(x_validation) ) + sum ( is.na(y_training) ) + sum(  is.na(y_validation) )

  normalizer <- create.package.normalize.list(s.x, m.x, y_max = y_max, y_min = y_min)




  probs <- seq(0, 1, length.out = n.knots)[-c(1, n.knots)]
  knots = quantile(rbind(y_training, y_validation),probs=seq(1/(n.knots-2), 1-1/(n.knots-2), length=n.knots-3))


  i_basis_training <- t(basis(y_training, n.knots,knots, integral = TRUE))
  i_basis_validation <- t(basis(y_validation, n.knots,knots, integral = TRUE))

  m_basis_training <- t(basis(y_training, n.knots, knots))
  m_basis_validation <- t(basis(y_validation, n.knots, knots))





  model <- in.fit.spqr(input_dim , hidden_dim , n.knots = n.knots , knots = knots, x_training = x_training,
                              x_validation = x_validation, y_training = y_training, y_validation = y_validation,
                              i_basis_training = i_basis_training, i_basis_validation = i_basis_validation,
                              m_basis_training = m_basis_training, m_basis_validation = m_basis_validation)

  variable_names <- NULL
  if (!is.null(colnames(x_training))) {
    variable_names <- colnames(x_training)
  }

  if (package.it) {
    if (!pre_normalize) {
      return (create.package.model(model = model, n.knots = n.knots, knots = knots,
                                   normalizer = normalizer, variable_names = variable_names, spqrx = FALSE))
    }
    return (create.package.model(model = model, n.knots = n.knots, knots = knots,
                                 normalizer = normalizer, variable_names = variable_names, spqrx = FALSE))
  }else {
    return (model)
  }

}






#' @export
fit.spqrx <- function(input_dim, hidden_dim, n.knots, x_training, x_validation, y_training, y_validation,
                      y.seq,hyperparameter = NULL, package.it = TRUE, pre_normalize = FALSE,
                      spqr_only = TRUE )
{


  if (!is.null(hyperparameter)) {

    p_a <- hyperparameter$p_a
    p_b <- hyperparameter$p_b
    c1 <- hyperparameter$c1
    c2 <- hyperparameter$c2


  }







  tf <- get("tf", envir = asNamespace("SPQRX"))
  y_max <- max(max(y_training), max(y_validation))
  y_min <- min(min(y_training), min(y_validation))

  y_training <- (y_training - y_min) /(y_max - y_min)
  y_validation <- (y_validation - y_min) / (y_max - y_min)


  x_combined <- rbind(x_training, x_validation)
  m.x <- apply(x_combined, 2, mean)
  s.x <- apply(x_combined, 2, sd)

  x_training <- scale(x_training, center = m.x, scale = s.x)
  x_validation <- scale(x_validation, center = m.x, scale = s.x)


  nas_sum <- sum ( is.na(x_training) )  + sum ( is.na(x_validation) ) + sum ( is.na(y_training) ) + sum(  is.na(y_validation) )
  print(paste0('Nas : ', nas_sum))

  normalizer <- create.package.normalize.list(s.x, m.x, y_max = y_max, y_min = y_min)




  probs <- seq(0, 1, length.out = n.knots)[-c(1, n.knots)]
  knots = quantile(rbind(y_training, y_validation),probs=seq(1/(n.knots-2), 1-1/(n.knots-2), length=n.knots-3))


  i_basis_training <- t(basis(y_training, n.knots,knots, integral = TRUE))
  i_basis_validation <- t(basis(y_validation, n.knots,knots, integral = TRUE))

  m_basis_training <- t(basis(y_training, n.knots, knots))
  m_basis_validation <- t(basis(y_validation, n.knots, knots))



  n.seq = 1001
  y.seq <- seq(0,1,length=n.seq)


  F.basis.seq <- tf$constant(basis(y.seq , n.knots,knots,integral = TRUE),dtype = 'float32') #this is used later to get quantiles
  f.basis.seq <- tf$constant(basis(y.seq , n.knots,knots),dtype = 'float32')
  y.seq <- tf$constant(y.seq, dtype = 'float32')

  model.heavy <- in.fit.spqrx(input_dim , hidden_dim , n.knots = n.knots , knots = knots, x_training = x_training,
                              x_validation = x_validation, y_training = y_training, y_validation = y_validation,
                              i_basis_training = i_basis_training, i_basis_validation = i_basis_validation,
                              m_basis_training = m_basis_training, m_basis_validation = m_basis_validation,
                              y.seq = y.seq, F.basis.seq = F.basis.seq, f.basis.seq = f.basis.seq,p_a = p_a, p_b = p_b ,c1 = c1 , c2 = c2 )

  variable_names <- NULL
  if (!is.null(colnames(x_training))) {
    variable_names <- colnames(x_training)
  }

  if (package.it) {
    if (!pre_normalize) {
      return (create.package.model(model = model.heavy, n.knots = n.knots, knots = knots,
                                   p_a = p_a, p_b = p_b,c1 = c1 , c2 = c2, normalizer = normalizer, variable_names = variable_names))
    }
    return (create.package.model(model = model.heavy, n.knots = n.knots, knots = knots, p_a = p_a, p_b = p_b , c1 = c1 , c2 = c2 , normalizer = normalizer,
                                 spqrx = TRUE))
  }else {
    return (model.heavy)
  }

}





#' @export
in.fit.spqrx <- function(input_dim, hidden_dim, n.knots,knots, x_training, x_validation, y_training, y_validation,
                         m_basis_training, m_basis_validation, i_basis_training, i_basis_validation,
                         y.seq , F.basis.seq, f.basis.seq, p_a = p_a, p_b = p_b,c1, c2)
{


  tf <- get("tf", envir = asNamespace("SPQRX"))

  model <- SPQRX(input_dim, hidden_dim, n.knots)


  model |> keras3::compile(
    loss = nloglik_loss_SPQR,
    optimizer = keras3::optimizer_adam(learning_rate = 0.001)
  )

  c1 <- max(c1, 25)
  checkpoint <- keras3::callback_model_checkpoint(filepath=paste0('runs/','model','/spqr_initial.weights.h5'), monitor = "val_loss", verbose = 0,
                                          save_best_only = TRUE, save_weights_only = TRUE, mode = "min",
                                          save_freq = "epoch")




  history <- model |> keras3::fit(
    list(covariates = x_training, data = y_training, I_basis = i_basis_training),
    m_basis_training,
    epochs = 200,
    batch_size = 32,
    callbacks=list(checkpoint,keras3::callback_early_stopping(monitor = "val_loss",
                                                      min_delta = 0, patience = 5)),
    validation_data=list(
      list(covariates = x_validation, data = y_validation, I_basis = i_basis_validation),
      m_basis_validation)
  )


  model.heavy <- SPQRX(input_dim = input_dim, hidden_dim = hidden_dim, k = n.knots)

  model.heavy <- keras3::load_model_weights(model.heavy,paste0('runs/','model','/spqr_initial.weights.h5'))


  # Consider setting c1 much higher than c2; it will give you positive densities



  model.heavy |> keras3::compile(
    loss = nloglik_loss(F.basis.seq,
                        f.basis.seq,
                        y.seq=y.seq,
                        p_a=p_a,p_b=p_b,
                        c1=c1,c2=c2,
                        lambda=NULL),
    optimizer = keras3::optimizer_adam()
  )

  checkpoint <- keras3::callback_model_checkpoint(filepath=paste0('runs/','model','/spqr_gpd.weights.h5'), monitor = "val_loss", verbose = 0,
                                          save_best_only = TRUE, save_weights_only = TRUE, mode = "min",
                                          save_freq = "epoch")

  history <- model.heavy |> keras3::fit(
    list(covariates = x_training, data = y_training, I_basis = i_basis_training),
    m_basis_training,
    epochs = 500,
    batch_size = 500,
    callbacks=list(checkpoint,keras3::callback_early_stopping(monitor = "val_loss",
                                                      min_delta = 0, patience = 20)),
    validation_data=list(
      list(covariates = x_validation, data = y_validation, I_basis = i_basis_validation),
      m_basis_validation)
  )


  #
  return (model.heavy)

}


#' @export
predict_spqrx<- function(object, x, y = NULL , type = 'QF', tau = 0.5, normalize_input = FALSE, normalize_output = TRUE)
{

  model <- object
  tf <- get("tf", envir = asNamespace("SPQRX"))
  if(!normalize_input) {

    y_max <- model$normalizer$y_max
    y_min <- model$normalizer$y_min
    m.x <- model$normalizer$x_mean
    s.x <- model$normalizer$x_std

    if (!is.null(y)) y <- (y - y_min) / (y_max - y_min)

      x <- scale(x, m.x, s.x)

    }

  # SPQR section of the code
  if (object$spqrx == FALSE) {
    knots <- model$knots

    y_max <- model$normalizer$y_max
    y_min <- model$normalizer$y_min
    n.knots <- (length(knots) + 3)
    model <- model$model


    if (type == 'CDF') {

      i_basis <- t(basis(y, n.knots, knots , integral = TRUE))
      m_basis <- t(basis(y, n.knots, knots , integral = FALSE))




      returnBack <- predict.spqrk(model = model, type = type, Y = y, knots = knots,
                                      I_basis = i_basis, covariates = x)

      }else if(type == 'PDF') {


        i_basis <- t(basis(y, n.knots, knots , integral = TRUE))
        m_basis <- t(basis(y, n.knots, knots , integral = FALSE))


        returnBack <- predict.spqrk(model = model, type = type, Y=y, knots = knots,I_basis = i_basis, covariates = x)

        return (returnBack)

      }else if(type == 'QF'){

        # Basis for quantile is not useful or used.


        if (is.vector(tau) && is.atomic(tau)) {


          i_basis <- matrix(0 , nrow = dim(x)[1], ncol = n.knots)
          m_basis <- matrix(0 , nrow = dim(x)[1], ncol = n.knots)
          returnBack = NULL
          for (index in 1:length(tau)) {



            temp_returnBack <- predict.spqrk(model = model, type = type, Y=NULL, knots = knots, I_basis = i_basis, covariates = x, tau = tau[index])

            if(is.null(returnBack)) {

              returnBack <- temp_returnBack
            }else {
              returnBack <- cbind(returnBack, temp_returnBack)
            }

          }

          colnames(returnBack) <- paste0((tau * 100) ," %")
          if (normalize_output) {
            returnBack <- (returnBack * (y_max - y_min)) + y_min
            return (returnBack)
          }else{
            return (returnBack)
          }
        }else {

          i_basis <- matrix(0 , nrow = dim(x)[1], ncol = n.knots)
          m_basis <- matrix(0 , nrow = dim(x)[1], ncol = n.knots)


          returnBack <- predict.spqrk(model = model, type = type, Y=NULL, knots = knots, I_basis = i_basis, covariates = x, tau = tau)

          if (normalize_output) {
            returnBack <- (returnBack * (y_max - y_min)) + y_min
            return (returnBack)
          }else{
            return (returnBack)
          }


        }



      }

    }

  model.heavy <- model$model
  p_a <- model$p_a
  p_b <- model$p_b
  c1 <- model$c1
  c2 <- model$c2
  knots <- model$knots

  y_max <- model$normalizer$y_max
  y_min <- model$normalizer$y_min
  n.knots <- (length(knots) + 3)


  if (type == 'CDF') {

    i_basis <- t(basis(y, n.knots, knots , integral = TRUE))
    m_basis <- t(basis(y, n.knots, knots , integral = FALSE))




    returnBack <- predict.spqrk.GPD(model = model.heavy, type = type, Y = y, knots = knots,
                                    I_basis = i_basis, M_basis = m_basis, covariates = x, p_a = p_a, p_b = p_b,  c1 = c1, c2 = c2)

  }else if(type == 'PDF') {


    i_basis <- t(basis(y, n.knots, knots , integral = TRUE))
    m_basis <- t(basis(y, n.knots, knots , integral = FALSE))


    returnBack <- predict.spqrk.GPD(model = model.heavy, type = type, Y=y, knots = knots,I_basis = i_basis,
                                    M_basis = m_basis, covariates = x, p_a = p_a, p_b = p_b,  c1 = c1, c2 = c2)

    return (returnBack)

  }else if(type == 'QF'){

    # Basis for quantile is not useful or used.


    if (is.vector(tau) && is.atomic(tau)) {


      i_basis <- matrix(0 , nrow = dim(x)[1], ncol = n.knots)
      m_basis <- matrix(0 , nrow = dim(x)[1], ncol = n.knots)
      returnBack = NULL
      for (index in 1:length(tau)) {



        temp_returnBack <- predict.spqrk.GPD(model = model.heavy, type = type, Y=NULL, knots = knots, I_basis = i_basis,
                                        M_basis = m_basis, covariates = x, p_a = p_a, p_b = p_b,  c1 = c1, c2 = c2, tau = tau[index])

        if(is.null(returnBack)) {

          returnBack <- temp_returnBack
        }else {
          returnBack <- cbind(returnBack, temp_returnBack)
        }

      }

      colnames(returnBack) <- paste0((tau * 100) ," %")
      if (normalize_output) {
        returnBack <- (returnBack * (y_max - y_min)) + y_min
        return (returnBack)
      }else{
        return (returnBack)
      }
    }else {

      i_basis <- matrix(0 , nrow = dim(x)[1], ncol = n.knots)
      m_basis <- matrix(0 , nrow = dim(x)[1], ncol = n.knots)


      returnBack <- predict.spqrk.GPD(model = model.heavy, type = type, Y=NULL, knots = knots, I_basis = i_basis,
                                      M_basis = m_basis, covariates = x, p_a = p_a, p_b = p_b,  c1 = c1, c2 = c2, tau = tau)

      if (normalize_output) {
        returnBack <- (returnBack * (y_max - y_min)) + y_min
        return (returnBack)
      }else{
        return (returnBack)
      }


    }



  }





}






eval.explain.ALE<- function(model,
                                   x,
                                   tau = 0.5,
                                   k = NULL,
                                   knots = NULL,
                                   var.index = c(1))
{
  tf <- get("tf", envir = asNamespace("SPQRX"))
  X <- x
  N <- nrow(X)
  d <- ncol(X)
  J <- var.index
  ntau <- length(tau)
  knots <- model$knots
  k <- (length(knots) + 3)


  firstcheck <- class(X[, J[1]]) == "numeric" ||
    class(X[, J[1]]) == "integer"

  if (!firstcheck)
    stop("X[,var.index] must be numeric or integer")


  z <- c(min(X[,J]), as.numeric(quantile(X[,J],seq(1/k,1,length.out=k), type=1)))

  z <- unique(z)
  k <- length(z) - 1

  f3 <- numeric(k)


  #a1 <- as.numeric(cut(X[, J], breaks = z, include.lowest = TRUE))

  a1 <- as.numeric(cut(X[, J], breaks = z, include.lowest = TRUE, labels = 1:k))


  X1 <- X
  X2 <- X
  X1[, J] <- z[a1]
  X2[, J] <- z[a1 + 1]




  y.hat1 <- predict_spqrx(
    object = model,
    x = X1,
    type = 'QF',
    tau = tau,
    normalize_output = FALSE
  )

  y.hat2 <- predict_spqrx(
    object = model,
    x = X2,
    type = 'QF',
    tau = tau,
    normalize_output = FALSE
  )



  Delta <- y.hat2 - y.hat1
  if (is.null(dim(Delta)))
    dim(Delta) <- c(N, 1)


  DDelta <- matrix(0, nrow = k, ncol = ntau)

  for (i in 1:ntau) {

    DDelta[,i] <- as.numeric(tapply(Delta[,i], a1, mean)) #K-length vector of averaged local effect values
  }

  f3 <- rbind(0, apply(DDelta, 2 , cumsum))

  return(list(x = z, ALE = f3))

}





#' @export
eval.explain.VI <- function(model, x, tau = seq(0.1, 0.9, 0.1),var.indexs = c(1, 2))
{

  varmatrix <- NULL

  for (var.index in var.indexs)  {

    result <- eval.explain.QALE(model, x, tau = tau, var.index = var.index)


    x_vals <- result$x
    f3 <- result$ALE

    variable.importance <- apply(f3, 2 , sd)



    if (is.null(varmatrix))
    {
      varmatrix <- variable.importance
    }else {
      varmatrix <- rbind(varmatrix, variable.importance)
    }


  }

  if (is.vector(varmatrix)) {
    varmatrix <- matrix(varmatrix, nrow = 1)
  }

  var.names <- paste0("variable_", var.indexs)
  rownames(varmatrix) <- var.names
  return (varmatrix)

}








#' @export
eval.explain.shapr <- function(model , x_training, x_explain, y_training, y_explain, type = 'QF', tau = 0.5,
                               variable_names = NULL)
{
  tf <- get("tf", envir = asNamespace("SPQRX"))

  if (is.null(variable_names)) {
    # The shapr library requires variable names for training
    colnames(x_training) <- paste0("variable_", 1:ncol(x_training))
    colnames(x_explain) <- paste0("variable_", 1:ncol(x_explain))
  }

  .shap.predict <- function(object, newdata, ...) {

    if (type == 'QF') {
      returnBack <- predict_spqrx(object, newdata, NULL, type = 'QF', tau = tau)
      return  (returnBack)
    } else if (type == 'CDF') {
      returnBack <- predict_spqrx(object, newdata, y_explain, type = 'CDF')
      return (returnBack)
    } else if (type == 'PDF') {
      returnBack <- predict_spqrx(object, newdata, y_explain, type = 'PDF')
      return (returnBack)
    }

  }


  if(type == 'QF'){


    quantile_mean <- predict_spqrx(model, x = x_training, type = 'QF', tau = tau)
    quantile_mean <- mean(quantile_mean)


    shapley_explantion <- shapr::explain(
      model = model,
      x_train = x_training,
      x_explain = x_explain,
      approach = "empirical",
      phi0 = quantile_mean,
      predict_model = .shap.predict,
    )

    return (shapley_explantion)

  }else if (type == 'CDF') {

    cdf_values <- predict.spqrx(model, x_training, y_training, type = 'CDF')
    mean_cdf <- mean(cdf_values)

    shapley_values <- shapr::explain(
        model = model,
        x_train = x_training,
        x_explain = x_explain,
        approach = "empirical",
        phi0 = mean_cdf,
        predict_model = .shap.predict,
      )

    return (shapley_values)
  }



  return (shapley_explantion)



}

#' @export
eval.explain.QALE <- function(model,
                                   x,
                                   tau = seq(0.1, 0.9, 0.1),
                                   var.index = c(1))
{


  X <- x
  N <- nrow(X)
  d <- ncol(X)
  J <- var.index
  ntau <- length(tau)

  knots <- model$knots
  k <- length(knots) + 3


  firstcheck <- class(X[, J[1]]) == "numeric" ||
    class(X[, J[1]]) == "integer"

  if (!firstcheck)
    stop("X[,var.index] must be numeric or integer")


  z <- c(min(X[,J]), as.numeric(quantile(X[,J],seq(1/k,1,length.out=k), type=1)))

  z <- unique(z)

  k <- length(z) - 1
  f3 <- numeric(k)


  #a1 <- as.numeric(cut(X[, J], breaks = z, include.lowest = TRUE))

  a1 <- as.numeric(cut(X[, J], breaks = z, include.lowest = TRUE, labels = 1:k))


  X1 <- X
  X2 <- X
  X1[, J] <- z[a1]
  X2[, J] <- z[a1 + 1]





  y.hat1 <- predict_spqrx(
    object = model,
    x = X1,
    type = 'QF',
    tau = tau,
    normalize_output = FALSE
  )

  y.hat2 <- predict_spqrx(
    object = model,
    x = X2,
    type = 'QF',
    tau = tau,
    normalize_output = FALSE
  )



  Delta <- y.hat2 - y.hat1
  if (is.null(dim(Delta)))
    dim(Delta) <- c(N, 1)


  DDelta <- matrix(0, nrow = k, ncol = ntau)

  for (i in 1:ntau) {

    DDelta[,i] <- as.numeric(tapply(Delta[,i], a1, mean)) #K-length vector of averaged local effect values
  }

  f3 <- rbind(0, apply(DDelta, 2 , cumsum))

  return(list(x = z, ALE = f3))

}


#' @export
#' @importFrom lime model_type predict_model
model_type.spqrx_model <- function(x, ...) {
  "regression"
}

#' @export
#' @importFrom lime model_type predict_model
predict_model.spqrx_model <- function(object, newdata, ...) {

  newdata <- as.data.frame(newdata)
  newdata <- as.matrix(newdata)

  preds <- predict_spqrx(
    object = object,
    x = newdata,
    type = "QF",
    tau = 0.5,
    normalize_input = FALSE
  )

  data.frame(Response = as.numeric(preds))
}

#' @export
eval.explain.lime <- function(model,
                              x_training,
                              x_explain,
                              tau = 0.5,
                              n_features = 5,
                              n_permutations = 5000)
{
  if (!requireNamespace("lime", quietly = TRUE)) {
    stop("Package 'lime' is required.")
  }


  x_training_norm <- scale(
    x_training,
    center = model$normalizer$x_mean,
    scale  = model$normalizer$x_std
  )

  x_explain_norm <- scale(
    x_explain,
    center = model$normalizer$x_mean,
    scale  = model$normalizer$x_std
  )

  x_training_norm <- as.data.frame(x_training_norm)
  x_explain_norm  <- as.data.frame(x_explain_norm)


  if(!is.null(model$variable_names)) {
    colnames(x_training_norm) <- model$variable_names
    colnames(x_explain_norm)  <- colnames(x_training_norm)
  }else {
    colnames(x_training_norm) <- paste0("V", seq_len(ncol(x_training_norm)))
    colnames(x_explain_norm)  <- colnames(x_training_norm)
  }

  predict_model.spqrx_model <- function(object, newdata, ...) {

    preds <- predict_spqrx(
      object = object,
      x = as.matrix(newdata),
      type = "QF",
      tau = tau,
      normalize_input = TRUE
    )

    data.frame(Response = as.numeric(preds))
  }

  explainer <- lime::lime(
    x = x_training_norm,
    model = model
  )

  explanation <- lime::explain(
    x_explain_norm,
    explainer,
    n_features = n_features,
    n_permutations = n_permutations
  )

  return(explanation)
}







#' @export
eval.plot.QVI <- function(model, x, var.indexs, lower_quantile = 0.1, upper_quantile = 0.9,
                          quantile_increment = 0.1)

{

  ALE_variable_estimates <- eval.explain.VI(model.heavy, x_testing,tau = seq(0.1, 0.9, 0.1), var.indexs = var.indexs)

  colnames(ALE_variable_estimates) <- paste((seq(0.1, 0.9, 0.1) * 100) , " %")

  x_vals <- 1:ncol(ALE_variable_estimates)
  matplot(
    x_vals,
    t(ALE_variable_estimates),    # transpose so each row becomes a line
    type = "l",
    lty = 1,
    lwd = 2,
    col = rainbow(nrow(ALE_variable_estimates)),
    xlab = "Covariate value",
    ylab = "Value",
    main = "Variable Effects",
    xaxt = "n"   # remove default axis
  )

  axis(
    1,
    at = 1:9,
    labels = paste0(seq(10, 90, 10), " %")
  )
  legend(
    "topleft",
    legend = rownames(ALE_variable_estimates),
    col = rainbow(nrow(ALE_variable_estimates) ),
    lty = 1,
    lwd = 2,
    cex = 0.8
  )


}






#' @export
eval.plot.lime <- function(model, x_training, x_explain, tau = 0.5)
{

  if(is.vector(x_explain)) {
    x_explain <- matrix(x_explain, nrow = 1)
  }


  lime_result <- eval.explain.lime(model, x_training, x_explain, tau = tau, n_features = ncol(x_training))



  lime_result <- dplyr::ungroup(
    dplyr::mutate(
      dplyr::group_by(lime_result, feature),
      mean_abs_weight = mean(abs(feature_weight))
    )
  )



  # Reorder features by mean absolute weight
  lime_result$feature <- reorder(
    lime_result$feature,
    lime_result$mean_abs_weight
  )



  # Plot
  ggplot2::ggplot(lime_result,
                  ggplot2::aes(
                    x = feature_weight,
                    y = feature,
                    color = as.numeric(feature_value)
                  )) +
    ggplot2::geom_jitter(
      height = 0.2,
      alpha = 0.6,
      size = 1.5
    ) +
    ggplot2::scale_color_viridis_c(option = "plasma") +
    ggplot2::labs(
      x = "LIME Contribution",
      y = "Feature",
      color = "Feature Value",
      title = "LIME Summary Plot"
    ) + ggplot2::theme_minimal()


}


#' @export
eval.plot.qexp <- function(model, x, y, pre_normalize = FALSE) {
  tf <- get("tf", envir = asNamespace("SPQRX"))
  cdf_values <- predict_spqrx(model, x, y, type = 'CDF')

  y_max <- model$normalizer$y_max
  y_min <- model$normalizer$y_min

  y <- (y - y_min) / (y_max - y_min )


  m_basis <- t( basis(y, (length(model$knots) + 3), model$knots) )

  qu <- stats::ppoints(max(1e3,length(m_basis)))


  qqplot(qexp(qu),qexp(cdf_values), xlim=range(qexp(cdf_values),qexp(qu)),ylim=range(qexp(cdf_values),qexp(qu)))
  abline(a = 0,b=1)


}


plot_spqrx_pdf <- function(model, x0, n_grid = 1000) {

  # Ensure single row
  if (is.vector(x0)) {
    x0 <- matrix(x0, nrow = 1)
  }

  # Extract normalization info
  y_min <- model$normalizer$y_min
  y_max <- model$normalizer$y_max

  # Grid in normalized space (model space)
  y_grid_norm <- seq(0, 1, length.out = n_grid)

  # Repeat covariate row for each y
  x_rep <- x0[rep(1, n_grid), , drop = FALSE]

  # Get normalized PDF
  pdf_norm <- predict_spqrx(
    object = model,
    x = x_rep,
    y = y_grid_norm,
    type = "PDF"
  )

  pdf_norm <- as.numeric(pdf_norm)

  # ---- CRITICAL STEP ----
  # Change of variables correction
  scale_factor <- 1 / (y_max - y_min)

  pdf_unnorm <- pdf_norm * scale_factor

  # Convert grid back to original scale
  y_grid <- y_grid_norm * (y_max - y_min) + y_min

  plot(
    y_grid,
    pdf_unnorm,
    type = "l",
    lwd = 2,
    xlab = "y",
    ylab = "Density",
    main = "Estimated PDF (Unnormalized Scale)"
  )

  invisible(list(y = y_grid, pdf = pdf_unnorm))
}












