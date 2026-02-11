

#' Custom activation function for SPQRX xi parameter
#'
#' Internal activation function used in the SPQRX neural network
#' architecture to constrain the xi parameter to the interval (0, 0.5).
#'
#' This function applies a sigmoid transformation and rescales
#' the output by 0.5. For the zeta parameter.
#'
#' @param x A tensor input from the previous neural network layer.
#'
#' @return A tensor with values constrained to (0, 0.5).
#'
#' @keywords internal
xi_custom_activation <- function(x) {
  0.5 * keras3::activation_sigmoid(x[keras3::all_dims(),1:1])#(0,0.5)
  # 0.5 * activation_tanh(x[all_dims(),1:1]) +0.1 #(-0.4,0.6)
}



#' Construct an SPQRX model object
#'
#' Internal constructor used to package a trained SPQR/SPQRX
#' model together with metadata required for prediction and
#' explainability methods.
#'
#' This function should not be called directly by users.
#'
#' @param model A trained keras model object.
#' @param n.knots Integer. Number of spline knots.
#' @param knots Numeric vector of knot locations.
#' @param hyperparameter Optional list containing model
#'   hyperparameters (e.g., p_a, p_b, c1, c2).
#' @param normalizer A list containing normalization parameters.
#' @param variable_names Optional character vector of covariate names.
#' @param spqrx Logical. TRUE if heavy-tail SPQRX model.
#'
#' @return An object of class `"spqrx_model"`.
#'
#' @keywords internal
create.package.model <- function(model, n.knots, knots, hyperparameter = NULL, normalizer = NULL,
                                 variable_names = NULL, spqrx = TRUE)
{
  return ( structure(
    list(model = model, n.knots = n.knots , knots = knots,
         hyperparameter = hyperparameter,
         normalizer = normalizer,
         variable_names = variable_names,
         spqrx = spqrx),
    class = "spqrx_model"
  ))
}


#' Construct normalization metadata
#'
#' Internal helper used to store normalization parameters
#' computed during model fitting.
#'
#' This function packages the scaling statistics required
#' to normalize inputs and rescale outputs during prediction.
#'
#' @param x_std Numeric vector of feature standard deviations.
#' @param x_mean Numeric vector of feature means.
#' @param y_max Maximum value of the response variable.
#' @param y_min Minimum value of the response variable.
#'
#' @return A list containing normalization parameters.
#'
#' @keywords internal
create.package.normalize.list <- function(x_std, x_mean, y_max, y_min)
{
  return (list(x_std = x_std, x_mean = x_mean, y_max = y_max, y_min = y_min))
}



#' Create SPQRX hyperparameter specification
#'
#' Constructs a structured hyperparameter list for use with
#' \code{fit.spqrx()}. This function allows users to explicitly
#' control tail modeling and training-related parameters.
#'
#' @param p_a Lower tail probability threshold.
#' @param p_b Upper tail probability threshold.
#' @param c1 Tail regularization parameter controlling density stability.
#' @param c2 Tail penalty parameter.
#' @param batch_size Mini-batch size used during training.
#'
#' @return A list containing hyperparameters that can be passed to
#'   \code{fit.spqrx()}.
#'
#' @examples
#' hp <- create.package.hyperparameter(
#'   p_a = 0.1,
#'   p_b = 0.9,
#'   c1 = 50,
#'   c2 = 10,
#'   batch_size = 500
#' )
#'
#' model <- fit.spqrx(
#'   input_dim = 5,
#'   hidden_dim = c(64, 32),
#'   n.knots = 10,
#'   x_training = X_train,
#'   x_validation = X_val,
#'   y_training = y_train,
#'   y_validation = y_val,
#'   hyperparameter = hp
#' )
#'
#' @export
create.packages.hyperparameter <- function(p_a = 0.9, p_b = 0.99, c1 = 25, c2 = 5, epochs = 200,
                                           batch_size = 300, activation = 'relu') {
  return (list(p_a = p_a, p_b = p_b, c1 = c1, c2 = c2,epochs = epochs, batch_size = batch_size,
               activation = activation))
}



#' Preprocess data for SPQR/SPQRX modeling
#'
#' Splits data into training, validation, and testing sets and
#' optionally applies normalization to predictors and response.
#'
#' When normalization is enabled, predictors are standardized
#' using training-set statistics, and the response is rescaled
#' to the unit interval.
#'
#' @param x Numeric matrix of covariates.
#' @param y Numeric vector or matrix of response values.
#' @param n.knots Integer. Number of spline knots.
#' @param testing_ratio Proportion of data allocated to testing.
#' @param valid_ratio Proportion of training data allocated to validation.
#' @param normalize Logical. If TRUE, standardizes predictors and rescales response.
#'
#' @return A list containing:
#' \itemize{
#'   \item Training, validation, and testing splits
#'   \item Spline knot locations
#'   \item Normalization metadata (if applicable)
#' }
#'
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


#' Build SPQRX Neural Network Architecture
#'
#' Constructs the Keras model architecture for the heavy-tail
#' SPQRX (Spline-based Probabilistic Quantile Regression) model.
#'
#' The network maps covariates to spline mixture probabilities
#' and a tail parameter \eqn{\xi}. The model output concatenates:
#' \itemize{
#'   \item Mixture probabilities over spline basis functions
#'   \item Tail parameter \eqn{\xi}
#'   \item Observed response values
#'   \item Spline basis evaluations
#' }
#'
#' This function defines only the network architecture. Model
#' compilation, training, and loss specification are handled
#' separately.
#'
#' @param input_dim Integer. Dimension of the covariate input.
#' @param hidden_dim Integer vector. Number of units in each hidden layer.
#' @param k Integer. Number of spline basis functions.
#' @param activation Provide the activation function for the hidden layers.
#'
#' @return A \code{keras_model} object representing the SPQRX architecture.
#'
#' @keywords internal
SPQRX <- function(input_dim, hidden_dim, k, activation = 'relu') {

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








#' Fit Base SPQR Neural Network (Internal Training Routine)
#'
#' Internal training routine for the baseline SPQR (Spline-based
#' Probabilistic Quantile Regression) model. This function builds
#' the neural network architecture, compiles it with the SPQR
#' negative log-likelihood loss, and performs supervised training
#' using precomputed spline basis matrices.
#'
#' This function assumes that:
#' \itemize{
#'   \item Covariates and responses have already been preprocessed.
#'   \item Spline basis matrices (`m_basis_*`, `i_basis_*`) have
#'         already been constructed.
#'   \item Hyperparameters are supplied via a structured list.
#' }
#'
#' It is not intended to be called directly by end users.
#'
#' @param input_dim Integer. Number of covariates (input features).
#' @param hidden_dim Integer vector. Number of units in each hidden layer.
#' @param n.knots Integer. Total number of spline basis functions.
#' @param knots Numeric vector of spline knot locations.
#' @param x_training Numeric matrix of training covariates.
#' @param x_validation Numeric matrix of validation covariates.
#' @param y_training Numeric vector or matrix of training responses.
#' @param y_validation Numeric vector or matrix of validation responses.
#' @param m_basis_training Numeric matrix of spline basis evaluations
#'   for the training responses.
#' @param m_basis_validation Numeric matrix of spline basis evaluations
#'   for the validation responses.
#' @param i_basis_training Numeric matrix of integrated spline basis
#'   evaluations for the training responses.
#' @param i_basis_validation Numeric matrix of integrated spline basis
#'   evaluations for the validation responses.
#' @param hyperparameter List. Model training configuration created by
#'   \code{create.packages.hyperparameter()}, containing elements such
#'   as `epochs`, `batch_size`, and `activation`.
#'
#' @return A trained \code{keras_model} object corresponding to the
#'   fitted SPQR architecture.
#'
#' @details
#' The model is trained using the \code{nloglik_loss_SPQR} loss
#' function and the Adam optimizer. Early stopping is applied
#' based on validation loss, and the best-performing weights are
#' saved during training.
#'
#' @seealso \code{\link{fit.spqr}}, \code{\link{SPQRX}}
#' @keywords internal
in.fit.spqr <- function(input_dim, hidden_dim, n.knots,knots, x_training, x_validation, y_training, y_validation,
                        m_basis_training, m_basis_validation, i_basis_training, i_basis_validation, hyperparameter = NULL)
{


  if(is.null(hyperparameter)) {
    stop('You most define the hyperparameter')
  }


  epochs <- hyperparameter$epochs
  batch_size <- hyperparameter$batch_size
  activation <- hyperparameter$activation


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
    epochs = epochs,
    batch_size = batch_size,
    callbacks=list(checkpoint,keras3::callback_early_stopping(monitor = "val_loss",
                                                              min_delta = 0, patience = 5)),
    validation_data=list(
      list(covariates = x_validation, data = y_validation, I_basis = i_basis_validation),
      m_basis_validation)
  )



  return (model)

}



#' Fit SPQR Model
#'
#' Fits the baseline SPQR (Spline-based Probabilistic Quantile Regression)
#' model using a neural network parameterization of spline mixture weights.
#'
#' This function performs:
#' \itemize{
#'   \item Response rescaling to the unit interval
#'   \item Covariate standardization using combined training/validation statistics
#'   \item Construction of spline basis and integrated basis matrices
#'   \item Neural network training via \code{in.fit.spqr()}
#' }
#'
#' The fitted model can optionally be returned as a packaged
#' \code{"spqrx_model"} object containing normalization metadata
#' and spline information for downstream prediction and explainability.
#'
#' @param input_dim Integer. Number of covariates (input features).
#' @param hidden_dim Integer vector specifying hidden layer sizes.
#' @param n.knots Integer. Total number of spline basis functions.
#' @param x_training Numeric matrix of training covariates.
#' @param x_validation Numeric matrix of validation covariates.
#' @param y_training Numeric vector or matrix of training responses.
#' @param y_validation Numeric vector or matrix of validation responses.
#' @param hyperparameter List created by
#'   \code{create.packages.hyperparameter()} specifying training
#'   configuration (e.g., epochs, batch size).
#' @param pre_normalize Logical. Included for API consistency.
#'   Currently does not alter preprocessing behavior.
#' @param package.it Logical. If TRUE (default), returns a structured
#'   \code{"spqrx_model"} object. If FALSE, returns the raw
#'   \code{keras_model}.
#'
#' @return
#' If \code{package.it = TRUE}, returns an object of class
#' \code{"spqrx_model"} containing:
#' \itemize{
#'   \item The trained keras model
#'   \item Spline knot locations
#'   \item Normalization parameters
#'   \item Variable names (if available)
#' }
#'
#' Otherwise, returns a trained \code{keras_model} object.
#'
#' @details
#' The response variable is internally rescaled to the unit interval
#' prior to spline construction. Covariates are standardized using
#' mean and standard deviation computed from the combined
#' training and validation data.
#'
#' @seealso \code{\link{fit_spqrx}}, \code{\link{predict_spqrx}}
#' @keywords internal
fit.spqr <- function(input_dim, hidden_dim, n.knots, x_training, x_validation, y_training, y_validation,
                     hyperparameter = NULL,pre_normalize = FALSE, package.it = TRUE)
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
                              m_basis_training = m_basis_training, m_basis_validation = m_basis_validation,
                              hyperparameter = hyperparameter)

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

#' Fit SPQRX Model
#'
#' Trains a Semi-Parametric Quantile Regression with eXtreme modeling (SPQRX)
#' neural network using training and validation datasets. The function
#' standardizes covariates, normalizes the response variable to [0,1],
#' constructs spline basis representations, and fits the SPQRX architecture.
#'
#' @param input_dim Integer. Number of input covariates.
#' @param hidden_dim Integer or vector. Number of hidden units in the network.
#' @param n.knots Integer. Number of spline knots.
#' @param x_training Matrix or data frame of training covariates.
#' @param x_validation Matrix or data frame of validation covariates.
#' @param y_training Numeric vector of training responses.
#' @param y_validation Numeric vector of validation responses.
#' @param hyperparameter List. Optional hyperparameters including tail parameters
#'   (e.g., p_a, p_b, c1, c2).
#' @param package.it Logical. If TRUE, returns a packaged model object with
#'   normalization and knot information.
#' @param pre_normalize Logical. Indicates whether prediction functions should
#'   assume pre-normalized inputs.
#' @param pre_train Logical. If TRUE, performs pre-training of the heavy-tail
#'   component.
#'
#' @return A fitted SPQRX model object. If `package.it = TRUE`, returns a
#'   packaged model including normalization parameters, knots, and metadata.
#'
#' @keywords internal
fit.spqrx <- function(input_dim, hidden_dim, n.knots, x_training, x_validation, y_training, y_validation,
                      hyperparameter = NULL, package.it = TRUE, pre_normalize = FALSE, pre_train = TRUE)
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
                              y.seq = y.seq, F.basis.seq = F.basis.seq, f.basis.seq = f.basis.seq,
                              hyperparameter = hyperparameter, pre_train = pre_train )

  variable_names <- NULL
  if (!is.null(colnames(x_training))) {
    variable_names <- colnames(x_training)
  }

  if (package.it) {
    if (!pre_normalize) {
      return (create.package.model(model = model.heavy, n.knots = n.knots, knots = knots, hyperparameter = hyperparameter, normalizer = normalizer, variable_names = variable_names))
    }
    return (create.package.model(model = model.heavy, n.knots = n.knots, knots = knots, hyperparameter = hyperparameter , normalizer = normalizer,
                                 spqrx = TRUE))
  }else {
    return (model.heavy)
  }

}


#' Internal SPQRX Training Routine
#'
#' Internal function used by \code{fit.spqrx()} to train the SPQRX model.
#' Handles optional pre-training of the spline component followed by
#' heavy-tail (GPD) optimization.
#'
#' @param input_dim Integer. Number of input covariates.
#' @param hidden_dim Integer or vector. Hidden layer configuration.
#' @param n.knots Integer. Number of spline knots.
#' @param knots Numeric vector of spline knot locations.
#' @param x_training Matrix of training covariates.
#' @param x_validation Matrix of validation covariates.
#' @param y_training Numeric vector of training responses (normalized).
#' @param y_validation Numeric vector of validation responses (normalized).
#' @param m_basis_training Matrix of spline basis evaluations for training data.
#' @param m_basis_validation Matrix of spline basis evaluations for validation data.
#' @param i_basis_training Matrix of integrated spline basis evaluations (training).
#' @param i_basis_validation Matrix of integrated spline basis evaluations (validation).
#' @param y.seq Tensor sequence over [0,1] used for density and quantile computation.
#' @param F.basis.seq Tensor of integrated spline basis evaluated over \code{y.seq}.
#' @param f.basis.seq Tensor of spline basis evaluated over \code{y.seq}.
#' @param hyperparameter List containing training configuration and tail parameters
#'   (e.g., epochs, batch_size, activation, p_a, p_b, c1, c2).
#' @param pre_train Logical. If TRUE, performs initial SPQR pre-training before
#'   heavy-tail optimization.
#'
#' @return A trained keras model object representing the SPQRX model.
#'
#' @keywords internal
in.fit.spqrx <- function(input_dim, hidden_dim, n.knots,knots, x_training, x_validation, y_training, y_validation,
                                 m_basis_training, m_basis_validation, i_basis_training, i_basis_validation
                                 ,y.seq , F.basis.seq, f.basis.seq, hyperparameter, pre_train = TRUE)
{


  tf <- get("tf", envir = asNamespace("SPQRX"))




  # Fetching parameter
  epochs <- hyperparameter$epochs
  batch_size <- hyperparameter$batch_size
  activation <- hyperparameter$activation

  p_a <- hyperparameter$p_a
  p_b <- hyperparameter$p_b
  c1 <- hyperparameter$c1
  c2 <- hyperparameter$c2


  # Checking Condition
  c1 <- max(c1, 25)





  if (pre_train ==TRUE ) {




    model <- SPQRX(input_dim, hidden_dim, n.knots)


    model |> keras3::compile(
      loss = nloglik_loss_SPQR,
      optimizer = keras3::optimizer_adam(learning_rate = 0.001)
    )

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

  }else {


    model <- SPQRX(input_dim = input_dim, hidden_dim = hidden_dim, k = n.knots,activation = activation)



    model |> keras3::compile(
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

    history <- model |> keras3::fit(
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
    return (model)

  }

}


#' Unified SPQR / SPQRX Model Fitting Interface
#'
#' High-level wrapper for fitting either the SPQR or SPQRX model.
#' This function selects the appropriate fitting routine based on
#' the \code{spqrx} flag and optionally performs pre-training for
#' the SPQRX model.
#'
#' @param input_dim Integer. Number of input covariates.
#' @param hidden_dim Integer or vector. Hidden layer configuration.
#' @param n.knots Integer. Number of spline knots.
#' @param x_training Matrix or data frame of training covariates.
#' @param x_validation Matrix or data frame of validation covariates.
#' @param y_training Numeric vector of training responses.
#' @param y_validation Numeric vector of validation responses.
#' @param hyperparameter List of hyperparameters. If \code{NULL},
#'   default values are generated via \code{create.packages.hyperparameter()}.
#' @param package.it Logical. Passed to lower-level fitting routines to
#'   determine whether the model should be returned as a packaged object. If false, passes back
#'   a trained keras object.
#' @param pre_normalize Logical. Indicates whether prediction functions
#'   should assume pre-normalized inputs.
#' @param spqrx Logical. If TRUE (default), fits the SPQRX model.
#'   If FALSE, fits the baseline SPQR model.
#' @param pre_train Logical. If TRUE, performs pre-training when
#'   fitting the SPQRX model.
#'
#' @return A fitted SPQR or SPQRX model object, depending on the
#'   \code{spqrx} flag.
#'
#' @export
fit_spqrx <- function(input_dim, hidden_dim, n.knots, x_training, x_validation, y_training, y_validation,
                       hyperparameter = NULL, package.it = TRUE, pre_normalize = FALSE, spqrx = TRUE, pre_train = TRUE)
{

  if (is.null(hyperparameter)) {
    # Just getting the default
    hyperparameter <- create.packages.hyperparameter()
  }else{


    if (!hyperparameter$activation %in% c("tanh", "relu", "sigmoid")) {
      hyperparameter = 'relu'
    }
  }

  if( spqrx == TRUE) {

    if (pre_train == TRUE){
      model.heavy <- fit.spqrx(input_dim , hidden_dim , n.knots , x_training, x_validation,
                         y_training, y_validation, hyperparameter = hyperparameter)

      return (model.heavy)
    }else{
      model.heavy <- fit.spqrx(input_dim , hidden_dim , n.knots, x_training,
                                       x_validation, y_training, y_validation, hyperparameter = hyperparameter, pre_train = FALSE)

      return (model.heavy)
    }

  }else {


    # There is no pre-training for regular SPQRX
    model <- fit.spqr(input_dim, hidden_dim, n.knots, x_training, x_validation,
                      y_training, y_validation, hyperparameter = hyperparameter)


    return (model)
  }




}


#' Predict from SPQR / SPQRX Model
#'
#' Generates predictions from a fitted SPQR or SPQRX model object.
#' Supports cumulative distribution (CDF), probability density (PDF),
#' and quantile function (QF) evaluation.
#'
#' @param object A fitted model object returned by \code{fit_spqrx()},
#'   \code{fit.spqrx()}, or \code{fit.spqr()}.
#' @param x Matrix or data frame of covariates for prediction.
#' @param y Optional numeric vector of response values. Required for
#'   \code{type = "CDF"} and \code{type = "PDF"}.
#' @param type Character string specifying prediction type:
#'   \code{"QF"} (quantile function), \code{"CDF"}, or \code{"PDF"}.
#' @param tau Numeric value or vector of quantile levels in (0,1).
#'   Used when \code{type = "QF"}.
#' @param normalize_input Logical. If FALSE (default), covariates and
#'   response values are normalized using stored model parameters.
#' @param normalize_output Logical. If TRUE (default), quantile predictions
#'   are transformed back to the original response scale.
#'
#' @return
#' \itemize{
#'   \item For \code{type = "QF"}: Matrix of predicted quantiles.
#'   \item For \code{type = "CDF"}: Matrix or vector of cumulative probabilities.
#'   \item For \code{type = "PDF"}: Matrix or vector of density values.
#' }
#'
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

      return (returnBack)

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

    }else{

      model.heavy <- model$model
      p_a <- model$hyperparameter$p_a
      p_b <- model$hyperparameter$p_b
      c1 <- model$hyperparameter$c1
      c2 <- model$hyperparameter$c2

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
            p_a <- model$hyperparameter$p_a
            p_b <- model$hyperparameter$p_b
            c1 <- model$hyperparameter$c1
            c2 <- model$hyperparameter$c2


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





}





#' Accumulated Local Effects (ALE) for SPQR / SPQRX Models
#'
#' Computes first-order Accumulated Local Effects (ALE) for a selected
#' covariate using a fitted SPQR or SPQRX model. The ALE function measures
#' the average change in the predicted quantile as the feature varies
#' across empirical intervals.
#'
#' @param model A fitted model object returned by \code{fit_spqrx()}.
#' @param x Matrix or data frame of covariates used to compute ALE.
#' @param tau Numeric vector of quantile levels in (0,1).
#' @param k Integer. Number of intervals used to partition the selected
#'   feature. If \code{NULL}, it is determined from the spline structure.
#' @param knots Numeric vector of spline knots. If \code{NULL},
#'   extracted from the fitted model.
#' @param var.index Integer vector specifying the column index of the
#'   covariate for which ALE is computed.
#'
#' @return A list containing:
#' \itemize{
#'   \item \code{x}: The interval boundary points used for the feature.
#'   \item \code{ALE}: Matrix of accumulated local effects. Rows correspond
#'   to feature intervals and columns correspond to quantile levels.
#' }
#'
#' @keywords internal
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





#' Quantile-Based Variable Importance for SPQR / SPQRX
#'
#' Computes variable importance across quantile levels using
#' Quantile Accumulated Local Effects (QALE). Importance is defined
#' as the standard deviation of the ALE function across feature intervals
#' for each quantile level.
#'
#' @param model A fitted model object returned by \code{fit_spqrx()}.
#' @param x Matrix or data frame of covariates used for evaluation.
#' @param tau Numeric vector of quantile levels in (0,1).
#' @param var.indexs Integer vector specifying the indices of variables
#'   for which importance is computed.
#'
#' @return A matrix of variable importance values.
#'   Rows correspond to variables and columns correspond to quantile levels.
#'
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
  colnames(varmatrix) <- paste0((tau * 100)," %")

  return (varmatrix)

}







#' Shapley Explanations for SPQR / SPQRX Models using shapr
#'
#' Computes Shapley values for SPQR or SPQRX model predictions
#' using the \code{shapr} package. Supports quantile (QF),
#' cumulative distribution (CDF), and density (PDF) explanations.
#'
#' @param model A fitted model object returned by \code{fit_spqrx()}.
#' @param x_training Matrix or data frame of training covariates.
#' @param x_explain Matrix or data frame of observations to explain.
#' @param y_training Numeric vector of training responses.
#' @param y_explain Numeric vector of response values used for
#'   CDF or PDF explanations.
#' @param type Character string specifying explanation target:
#'   \code{"QF"}, \code{"CDF"}, or \code{"PDF"}.
#' @param tau Numeric quantile level in (0,1) used when
#'   \code{type = "QF"}.
#' @param shapley_method Character string specifying the
#'   \code{shapr} estimation approach (e.g., \code{"empirical"}).
#' @param variable_names Optional character vector of variable names.
#'   If \code{NULL}, default names are assigned.
#' @param original_output Logical. If TRUE, returns the full
#'   \code{shapr} explanation object. If FALSE (default),
#'   returns a data frame of Shapley values.
#'
#' @return
#' If \code{original_output = FALSE}, a data frame of Shapley values.
#' Otherwise, the full \code{shapr} explanation object.
#'
#' @export
eval.explain.shapr <- function(model , x_training, x_explain, y_training, y_explain, type = 'QF', tau = 0.5, shapley_method = 'empirical',
                               variable_names = NULL, original_output = FALSE)
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


    shapley_explanation <- shapr::explain(
      model = model,
      x_train = x_training,
      x_explain = x_explain,
      approach = shapley_method,
      phi0 = quantile_mean,
      predict_model = .shap.predict,
    )


    if (!original_output) {

      shap_df <- as.data.frame(shapley_explanation$shapley_values_est)



      return(shap_df)

    } else {
      return(shapley_explanation)
    }


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

#' Quantile Accumulated Local Effects (QALE)
#'
#' Computes Quantile Accumulated Local Effects (QALE) for a selected
#' covariate using a fitted SPQR or SPQRX model. The function evaluates
#' changes in predicted quantiles across empirical feature intervals
#' and accumulates the averaged local effects.
#'
#' @param model A fitted model object returned by \code{fit_spqrx()}.
#' @param x Matrix or data frame of covariates.
#' @param tau Numeric vector of quantile levels in (0,1).
#' @param var.index Integer specifying the column index of the
#'   covariate to evaluate.
#'
#' @return A list containing:
#' \itemize{
#'   \item \code{x}: Interval boundary points for the selected feature.
#'   \item \code{ALE}: Matrix of accumulated local effects across quantiles.
#' }
#'
#' @keywords internal
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



#' LIME Explanations for SPQR / SPQRX Models
#'
#' Computes local feature attributions using the \code{lime} package
#' for quantile predictions from a fitted SPQR or SPQRX model.
#' Explanations are generated on standardized covariates using
#' the stored normalization parameters.
#'
#' @param model A fitted model object returned by \code{fit_spqrx()}.
#' @param x_training Matrix or data frame of training covariates.
#' @param x_explain Matrix or data frame of observations to explain.
#' @param tau Numeric quantile level in (0,1) for quantile prediction.
#' @param n_permutations Integer. Number of permutations used by
#'   \code{lime::explain()}.
#' @param original_output Logical. If TRUE, returns the full LIME
#'   explanation object. If FALSE (default), returns a reshaped
#'   feature-weight table.
#'
#' @return
#' If \code{original_output = FALSE}, a data frame where rows correspond
#' to explained cases and columns correspond to feature weights.
#' Otherwise, the full \code{lime} explanation object.
#'
#' @export
eval.explain.lime <- function(model,
                              x_training,
                              x_explain,
                              tau = 0.5,
                              n_permutations = 5000,
                              original_output = FALSE)
{


  n_features <- ncol(x_training)


  if (!requireNamespace("lime", quietly = TRUE)) {
    stop("Package 'lime' is required.")
  }


  p_a <- model$hyperparameter$p_a
  p_b <- model$hyperparameter$p_b
  c1 <- model$hyperparameter$c1
  c2 <- model$hyperparameter$c2


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


  if (!original_output) {
    library(dplyr)
    library(tidyr)

    lime_table <- explanation %>%
      select(case, feature, feature_weight) %>%
      pivot_wider(
        names_from = feature,
        values_from = feature_weight
      )

    return (lime_table)

  } else {

    return(explanation)

  }





}




#' Plot Quantile Variable Importance (QVI)
#'
#' This function generates a line plot of variable effects across quantiles
#' using the Quantile Variable Importance (QVI) calculated from a fitted SPQRX model.
#'
#' @param model A fitted SPQRX or SPQRx model object.
#' @param x A data frame or matrix of covariates used for evaluation.
#' @param var.indexs An integer vector specifying which covariates to plot.
#' @param lower_quantile Numeric. Lower quantile bound (default 0.1).
#' @param upper_quantile Numeric. Upper quantile bound (default 0.9).
#' @param quantile_increment Numeric. Step size between quantiles (default 0.1).
#'
#' @details
#' The function computes Quantile Variable Importance (QVI) for the selected
#' covariates and visualizes them using a line plot. Each line represents a covariate,
#' and the x-axis corresponds to quantiles (in percentages).
#'
#' @return
#' A plot is displayed. No object is returned.
#'
#' @examples
#' \dontrun{
#' eval.plot.QVI(model = fitted_model, x = x_test, var.indexs = c(1,2,3))
#' }
#'
#' @export
eval.plot.QVI <- function(model, x, var.indexs, lower_quantile = 0.1, upper_quantile = 0.9,
                          quantile_increment = 0.1)

{

  ALE_variable_estimates <- eval.explain.VI(model.heavy, x_testing, tau = seq(0.1, 0.9, 0.1), var.indexs = var.indexs)

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






#' Plot LIME Explanation Summary
#'
#' Generates a summary plot of LIME (Local Interpretable Model-agnostic
#' Explanations) feature contributions for one or more observations.
#' The plot displays feature contributions on the x-axis and features
#' ordered by mean absolute contribution on the y-axis. Points are
#' colored by the corresponding feature values.
#'
#' @param model A fitted model object returned by \code{fit_spqrx()}.
#' @param x_training Matrix or data frame of training covariates used to
#'   build the LIME explainer.
#' @param x_explain Matrix, data frame, or numeric vector representing
#'   the observation(s) to explain.
#' @param tau Numeric. Quantile level used for prediction when generating
#'   LIME explanations. Default is \code{0.5}.
#'
#' @return Invisibly returns a \code{ggplot} object showing the LIME
#'   contribution summary.
#'
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


#' Plot SHAP Values
#'
#' Generates a summary plot of SHAP values for a given model
#' across features using results from \code{eval.explain.shapr()}.
#'
#' @param model A fitted model object returned by \code{fit_spqrx()}.
#' @param x_training Matrix or data frame used to train the model.
#' @param x_explain Matrix or data frame of covariates to explain.
#' @param y_training Vector of responses corresponding to \code{x_training}.
#' @param y_explain Vector of responses corresponding to \code{x_explain}.
#' @param tau Numeric or vector of quantiles (default 0.5) for QF predictions.
#' @param shapley_method Character, the SHAP approximation method (default 'empirical').
#'
#' @return Invisibly returns \code{NULL}. Produces a ggplot2 plot of SHAP feature contributions.
#'
#' @export
eval.plot.shap <- function(model, x_training, x_explain, y_training, y_explain,
                           tau = 0.5, shapley_method = 'empirical') {

  if (is.vector(x_explain)) {
    x_explain <- matrix(x_explain, nrow = 1)
  }

  # Compute SHAP values
  shap_result <- eval.explain.shapr(
    model = model,
    x_training = x_training,
    x_explain = x_explain,
    y_training = y_training,
    y_explain = y_explain,
    tau = tau,
    shapley_method = shapley_method,
    original_output = TRUE
  )

  library(dplyr)
  library(tidyr)
  library(ggplot2)

  shap_result <- shap_result$shapley_values_est[,-1] %>%
    as.data.frame() %>%
    tibble::rownames_to_column("case") %>%
    pivot_longer(
      cols = -case,
      names_to = "feature",
      values_to = "feature_weight"
  )

  # Plot
  ggplot(shap_result, aes(x = feature_weight, y = feature)) +
    geom_jitter(height = 0.2, alpha = 0.6, size = 1.5) +
    labs(
      x = "SHAP Contribution",
      y = "Feature",
      title = "SHAP Summary Plot"
    ) +
    theme_minimal()
}




#' Exponential Q-Q Plot for SPQR / SPQRX Model
#'
#' Produces an exponential Q-Q plot using model-based CDF values.
#' The function evaluates the fitted conditional distribution and
#' compares transformed probabilities to the theoretical exponential
#' distribution.
#'
#' @param model A fitted model object returned by \code{fit_spqrx()}.
#' @param x Matrix or data frame of covariates.
#' @param y Numeric vector of observed response values.
#' @param pre_normalize Logical. If FALSE (default), response values
#'   are normalized using stored model parameters.
#'
#' @return Invisibly returns \code{NULL}. Produces a base R Q-Q plot.
#'
#' @export
eval.plot.qexp <- function(model, x, y, pre_normalize = FALSE) {
  tf <- get("tf", envir = asNamespace("SPQRX"))


  cdf_values <- predict_spqrx(model, x, y, type = 'CDF')


  eps <- 1e-6
  cdf_values <- pmin(pmax(cdf_values, eps), 1 - eps)


  y_max <- model$normalizer$y_max
  y_min <- model$normalizer$y_min
  y <- (y - y_min) / (y_max - y_min)


  m_basis <- t(basis(y, (length(model$knots) + 3), model$knots))

  qu <- stats::ppoints(max(1e3, length(m_basis)))


  qqplot(
    qexp(qu),
    qexp(cdf_values),
    xlim = range(qexp(qu), qexp(cdf_values), na.rm = TRUE),
    ylim = range(qexp(qu), qexp(cdf_values), na.rm = TRUE)
  )
  abline(a = 0, b = 1)
}




#' Plot Predicted Probability Density Function (PDF) for a Single Observation
#'
#' Generates a plot of the predicted probability density function (PDF) for a single observation
#' using a fitted SPQRX model. The x-axis is the response variable \code{y}, and the y-axis
#' represents the predicted density. The plot is restricted to the range between the 1st and 99th
#' predicted quantiles for better visualization.
#'
#' @param model A fitted SPQRX model object returned by \code{fit_spqrx()}.
#' @param x0 A numeric vector representing the covariate values for the observation to plot.
#' @param npdf_points Integer. Number of points to evaluate the PDF along the y-axis. Default is 500.
#'
#' @return Invisibly returns \code{NULL}. Produces a ggplot2 plot of the predicted PDF for the observation.
#'
#' @details
#' The function computes the predicted PDF over a grid of y-values from the model's
#' normalized output range. It scales the y-axis based on the model's predicted
#' 1st and 99th quantiles to focus on the main density mass.
#'
#' @examples
#' \dontrun{
#' # Plot PDF for the 102nd observation in the dataset
#' eval.plot.pdf(model = fitted_model, x0 = x_training[102, ])
#' }
#'
#' @export
eval.plot.pdf <- function(model, x0, npdf_points = 500)
{

    if (is.vector(x0)) {
      x0 <- matrix(x0, nrow = 1)
    }

    if(nrow(x0) > 1) {
      stop("Only one observation for eval.plot.pdf function")
    }

    y_max <- model$normalizer$y_max
    y_min <- model$normalizer$y_min
    y_seq <- seq(y_min, y_max, length.out = npdf_points)


    highest_quantile <- 0.99
    high_quantile <- predict_spqrx(model, matrix(x_training[102,], nrow = 1), tau = highest_quantile)
    lowest_quantile <- 0.01
    low_quantile <- predict_spqrx(model, matrix(x_training[102,], nrow = 1), tau = lowest_quantile)

    pdf_values = c()
    # Get PDF predictions
    for(y_value in y_seq) {
      pdf_values <- c(pdf_values, predict_spqrx(
       model,
       matrix(x_training[102, ], nrow = 1),
       y = matrix(y_value, ncol = 1),
       type = "PDF",
       normalize_output = TRUE
      ))
    }

    # If pdf_values is a vector, convert to matrix for consistent plotting
    if (is.null(dim(pdf_values))) {
      pdf_values <- matrix(pdf_values, ncol = 1)
    }

    # Prepare data for ggplot
    df <- data.frame(
      y = rep(y_seq, times = nrow(pdf_values)),
      PDF = as.vector(t(pdf_values)),
      Obs = rep(1:nrow(pdf_values), each = length(y_seq))
    )

    # Plot using ggplot
    ggplot2::ggplot(df, ggplot2::aes(x = y, y = PDF, group = Obs, color = factor(Obs))) +
    ggplot2::geom_line() +
    ggplot2::labs(
     x = "y",
     y = "PDF",
     title = "Predicted Probability Density Function"
    )  + ggplot2::theme(legend.position = "none") + ggplot2::coord_cartesian(xlim = c(low_quantile, high_quantile))


}
