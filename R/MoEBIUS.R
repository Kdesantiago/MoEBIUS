source("R/F_utils.R")
library(caret)

#' Initialize parameters for MoEBIUS
#'
#' This function initializes the parameters for the MoEBIUS Model.
#'
#' @param X A matrix of predictors.
#' @param y A response vector.
#' @param K The number of clusters for rows.
#' @param Q The number of clusters for columns.
#' @param learning_rate The learning rate for the optimization process (default is 1e-3).
#' @param iter_max The maximum number of iterations (default is 10).
#' @param init_type The initialization type, either 'Kmeans' or other (default is 'Kmeans').
#' @return A list of initialized parameters.
#' @examples
#' X <- matrix(rnorm(100), 10, 10)
#' y <- rnorm(10)
#' params <- initializationCCLBM(X, y, K = 2, Q = 3)
#' @export
initializationCCLBM <- function(X,y,K,Q,learning_rate=1e-3,iter_max=10, init_type='Kmeans'){
  params = list()
  params$X <- as.matrix(X)
  params$y <- y
  params$learning_rate <- learning_rate
  params$iter_max <- iter_max
  params$K <- K
  params$Q <- Q

  params$p <- ncol(params$X)
  params$N <- nrow(params$X)

  params$pi_k <- matrix(rnorm(params$p * K),params$p,K)
  params$beta <- array(rnorm(params$K*params$Q),dim=c(params$K,params$Q))


  if(init_type == "Kmeans"){
    tau <- one_hot_errormachine(tryCatch({ kmeans(params$X,centers=K,nstart = 50,iter.max = 10)$cluster},
                                         error=function(cond){
                                           #message("Kmeans with small noise")
                                           mat2 <- params$X + matrix(runif(n = nrow(params$X) * ncol(params$X),min=0,max=1e-8),nrow(params$X),ncol(params$X))
                                           km= kmeans(mat2,centers=K,nstart = 50)$cluster
                                           return(km)
                                         }))

    clust_tau = apply(tau,1,which.max)
    nu <- array(NA,dim = c(params$K,params$p,params$Q))
    for(k in 1:K){
      mat = t(params$X[clust_tau == k,])
      nu[k,,] <- one_hot_errormachine(tryCatch({ kmeans(mat,centers=Q,nstart = 50)$cluster},
                                               error=function(cond){
                                                 #message("Kmeans with small noise")
                                                 mat2 <- mat + matrix(runif(n = nrow(mat) * ncol(mat),min=0,max=.1e-8),nrow(mat),ncol(mat))
                                                 km= kmeans(mat2,centers=Q,nstart = 50)$cluster
                                                 return(km)
                                               }))
    }

    params$tau <- tau
    params$nu <- nu
  }
  return(params)
}

#' Update the \eqn{\pi_k} parameter for the MoEBIUS model
#'
#' Updates the matrix of row-cluster membership regression parameters \eqn{\pi_k} based on current estimates.
#'
#' @param params A list containing current model parameters.
#' @return The updated list of parameters including the modified \eqn{\pi_k}.
#' @export
update_pi_CCLBM <- function(params){

  X <- params$X
  y <- params$y
  tau <- params$tau
  nu <- params$nu
  pi_k <- params$pi_k
  learning_rate <- params$learning_rate

  grad_pi = array(NA,dim=c(params$p,params$K))

  N = nrow(tau)
  K = ncol(tau)
  Q = dim(nu)[3]

  # Pred Si_k (Zik)
  S_ik = Softmax_log(X %*% pi_k)

  grad_pi = t(X) %*% (tau - S_ik)

  params$pi_k <- pi_k + learning_rate/params$N * grad_pi # On prend la moyenne des gradients
  return(params)
}

#' Update row cluster membership probabilities (\eqn{\tau}) in MoEBIUS model
#'
#' This function updates the \eqn{\tau} parameter representing row-cluster probabilities.
#'
#' @param params A list of model parameters.
#' @param sampling Logical; if TRUE, uses sampling for the update (default is TRUE).
#' @return The updated list of parameters with the new \eqn{\tau}.
#' @export
update_tau_CCLBM <- function(params,sampling=TRUE){

  X <- params$X
  y <- params$y
  nu <- params$nu
  pi_k <- params$pi_k
  beta <- params$beta


  N <- length(y)
  K <- params$K
  p <- dim(nu)[2]
  Q <- dim(nu)[3]
  sigma2_k <- params$sigma2_k
  ## Densité y
  log_y_prob = array(NA,dim=c(N,K))
  for(k in 1:K){
    log_y_prob[,k]= log_err(dnorm(y,mean = X %*% nu[k,,] %*% beta[k,],sd = sqrt(sigma2_k[k])))
  }
  #log_y_prob = t(sapply(1:N,function(i){log_y_prob[,i,as.numeric(params$y)[i]]}))

  # Probas non normalisées
  Tik =  log_err(Softmax_log(X %*% pi_k)) + log_y_prob

  # Résultat
  res = Softmax_log(Tik)
  if(sampling){ res = one_hot_errormachine(
    sapply(1:params$N,function(i){sample(1:params$K,size = 1,prob = res[i,])}),size=params$K)}
  params$tau = pmax(res,.Machine$double.eps)

  return(params)
}

#' Update column cluster membership probabilities (\eqn{\rho}) in MoEBIUS model
#'
#' Updates the \eqn{\rho} parameter based on current estimates of \eqn{\nu}.
#'
#' @param params A list of model parameters.
#' @return The updated list of parameters with the modified \eqn{\rho}.
#' @export
update_rho_CCLBM <- function(params){
  nu <- params$nu
  p <- dim(nu)[2]
  res <- apply(nu,c(1,3),sum)/p
  params$rho_ks = pmax(res,.Machine$double.eps)
  return(params)
}

#' Update \eqn{\beta} parameter for the MoEBIUS model
#'
#' Updates \eqn{\beta}, the regression coefficients for each clusters.
#'
#' @param params A list of model parameters.
#' @return The updated list of parameters including the modified \eqn{\beta}.
#' @export
update_beta_CCLBM <- function(params){

  X <- params$X
  y <- params$y
  tau <- params$tau
  nu <- params$nu
  beta <- params$beta
  learning_rate <- params$learning_rate

  #grad_beta = array(NA,dim=c(params$K,params$Q))

  N = nrow(tau)
  K = ncol(tau)
  Q = dim(nu)[3]


  for(k in 1:K){
    X_tild <- t(nu[k,,]) %*% t(X) %*% diag(tau[,k])
    beta[k,] <- solve(X_tild %*% X %*% nu[k,,] + 1e-6 * diag(Q)) %*% X_tild %*% y
    #   grad_beta[k,,] = (t(X %*% nu[k,,]) * (matrix(1,params$Q,1) %*% matrix(tau[,k],1,params$N))) %*% (params$y_OHE - y_prob[k,,])
  }

  params$beta <- beta
  #params$beta <- beta + learning_rate/params$N * grad_beta # On prend la moyenne des gradients
  return(params)
}

#' Update variance parameters (\eqn{\sigma_k}) for each cluster in MoEBIUS model
#'
#' Calculates the variance within each row cluster based on current estimates.
#'
#' @param params A list of model parameters.
#' @return The updated list of parameters with the new \eqn{\sigma_k}.
#' @export
update_sigma_CCLBM <- function(params){
  X <- params$X
  y <- params$y
  tau <- params$tau
  nu <- params$nu
  beta <- params$beta
  learning_rate <- params$learning_rate
  K = params$K

  sigma_k <- rep(NA,K)

  for(k in 1:K){
    sigma_k[k] = t(tau[,k]) %*% ((y - X %*% nu[k,,] %*% beta[k,])**2)
  }
  params$sigma2_k <- sigma_k / colSums(tau)
  return(params)
}

#' Update the \eqn{\nu} parameter in Co-CoLBMoE model
#'
#' Updates column-cluster membership probabilities \eqn{\nu} for each row cluster.
#'
#' @param params A list of model parameters.
#' @param sampling Logical; if TRUE, uses sampling for the update (default is TRUE).
#' @return The updated list of parameters with modified \eqn{\nu}.
#' @export
update_nu_CCLBM2 <- function(params,sampling=TRUE){

  X <- params$X
  y <- params$y
  tau <- params$tau
  nu <- params$nu
  rho_ks <- params$rho_ks
  beta <- params$beta

  N <- dim(tau)[1]
  K <- dim(tau)[2]
  p <- params$p
  Q <- dim(rho_ks)[2]

  V_kjs = aperm(array(rep(log_err(rho_ks), p),dim=c(K,Q,p)),c(1,3,2)) #remet en dim KxpxQ pour rho_ks
  sigma2_k <- params$sigma2_k


  ## Densité y
  for(k in 1:K){
    for(j in 1:p){
      nu_tilde <- nu[k,,]
      for(s in 1:Q){
        nu_tilde[j,] <- 0
        nu_tilde[j,s] <- 1

        tmp = log_err(dnorm(y,mean = X %*% nu_tilde %*% beta[k,],sd = sqrt(sigma2_k[k])))
        #tmp = matrix(sapply(1:N,function(i){tmp[i,as.numeric(params$y)[i] ]}),N,1)
        V_kjs[k,j,s] = V_kjs[k,j,s] + t(tau[,k]) %*% tmp
      }
    }
  }

  # Application softmax
  res = aperm(array(sapply(1:K,function(k){Softmax_log(V_kjs[k,,])}),dim=c(p,Q,K)),c(3,1,2))

  if(sampling){
    nu_sampled = t(sapply(1:params$K,function(k){
      sapply(1:params$p,function(j){
        sample(1:params$Q,size = 1,prob = res[k,j,])})
    }))
    res = aperm(array(sapply(1:params$K,function(k){
      one_hot_errormachine(nu_sampled[k,],size=params$Q)}),dim=c(params$p,params$Q,params$K)),c(3,1,2))
  }

  params$nu = pmax(res,.Machine$double.eps)
  return(params)
}

#' Calculate the Evidence Lower Bound (ELBO) for MoEBIUS model
#'
#' This function computes the ELBO for the current model state.
#'
#' @param params A list of model parameters.
#' @return A numeric value representing the ELBO.
#' @export
ELBO_CCLBM <- function(params){
  X <- params$X
  y <- params$y
  tau <- params$tau
  nu <- params$nu
  pi_k <- params$pi_k
  rho_ks <- params$rho_ks
  beta <- params$beta
  sigma2_k <- params$sigma2_k

  N <- dim(tau)[1]
  K <- dim(tau)[2]
  p <- params$p
  Q <- dim(rho_ks)[2]

  elbo <- 0

  for(k in 1:K){
    elbo = elbo +  t(tau[,k]) %*% log_err(dnorm(y,mean = X %*% nu[k,,] %*% beta[k,],sd = sqrt(sigma2_k[k])))
  }

  elbo = elbo + sum(tau * log_err(Softmax_log(params$X %*% params$pi_k)))
  elbo = elbo - sum(tau * log_err(tau) )

  elbo = elbo - sum(nu * log_err(nu))
  elbo = elbo + sum(nu * aperm(array(rep(log_err(rho_ks), p),dim=c(K,Q,p)),c(1,3,2)))


  return(elbo)
}

#' Fit MoEBIUS model for regression, with fixed parameters
#'
#' Fits a supervised MoEBIUS model using an iterative update procedure, with fixed numbers of clusters and components.
#'
#' @param X A matrix of predictors.
#' @param y A response vector.
#' @param K Number of row clusters.
#' @param Q Number of column clusters.
#' @param learning_rate Learning rate for optimization (default is 1e-3).
#' @param iter_max Maximum number of iterations (default is 10).
#' @param init_type Initialization type (default is 'Kmeans').
#' @return A list of fitted model parameters.
#' \dontrun{@examples
#' X <- matrix(rnorm(100), 10, 10)
#' y <- rnorm(10)
#' params <- cocoLBMoE_reg_fixe(X, y, K = 2, Q = 3)
#' }
#' @export
MoEBIUS_reg_fixe <- function(X,y,K,Q,learning_rate=1e-3,iter_max=10, init_type='Kmeans'){

  # Supervised cocoLBM
  params= initializationCCLBM(X,y,K,Q,learning_rate,iter_max, init_type)
  n_iter = 1
  while(n_iter <= params$iter_max){
    #if(n_iter %% 10 == 0) {print(n_iter)}
    params <- update_pi_CCLBM(params)
    params <- update_rho_CCLBM(params)
    params = update_beta_CCLBM(params)
    params = update_sigma_CCLBM(params)
    params <- update_nu_CCLBM2(params)
    params <- update_tau_CCLBM(params)
    #print(ELBO_CCLBM(params))
    n_iter = n_iter+1
  }

  params$elbo <- ELBO_CCLBM(params)
  params$ICL <- ELBO_CCLBM(params) -1/2 * params$p*(params$K-1)*log(params$N) -
    1/2 * params$K*(params$Q-1)*log(params$p) -
    1/2 * params$K * params$Q * log(params$N)
  return(params)

}

#' Predict responses using fitted MoEBIUS model
#'
#' Generates predictions for new data based on a fitted MoEBIUS model.
#'
#' @param params A list of fitted model parameters.
#' @param X_test A matrix of new predictor data.
#' @return A vector of predicted responses.
#' @export
prediction_MoEBIUS_reg <- function(params, X_test){
  tau <- Softmax_log(X_test %*% params$pi_k)
  y_par_classe <- sapply(1:params$K, function(k){X_test %*% params$nu[k,,] %*% params$beta[k,]})
  y <- apply(tau * y_par_classe,1,sum)
  return(y)
}


#' Mixture Of Experts and BIclustering Unified Strategy (MoEBIUS) for Regression
#'
#' Fits a MoEBIUS model to the provided data, selecting the best model based on ELBO and ICL criteria.
#' Optionally, cross-validation is performed to choose the best hyperparameters.
#'
#' @param X A matrix of predictor variables.
#' @param y A vector of response variables.
#' @param K_set A set of values for the number of individual clusters.
#' @param Q_set A set of values for the number of variable components
#' @param learning_rate The learning rate for optimization (default is `1e-3`).
#' @param iter_max Maximum number of iterations for optimization (default is `10`).
#' @param init_type Initialization method (default is `'Kmeans'`).
#' @param ALL Logical, if `TRUE`, saves all models tested (default is `FALSE`).
#' @param Cross_val Logical, if `TRUE`, performs 5-fold cross-validation to select the best (K, Q) combination (default is `FALSE`).
#'
#' @return A list containing:
#' \item{best_ICL}{Model with the highest ICL criterion value.}
#' \item{best_ELBO}{Model with the highest ELBO criterion value.}
#' \item{model_CV}{Best model found using cross-validation (if `Cross_val = TRUE`).}
#' \item{save}{All models tested (if `ALL = TRUE`).}
#'
#' @examples
#' \dontrun{
#' model <- MoEBIUS_reg(X, y, K_set = 2:4, Q_set = 1:3, Cross_val = TRUE)
#' }
#' @export
MoEBIUS_reg <- function(X,y,K_set,Q_set,learning_rate=1e-3,iter_max=10, init_type='Kmeans',
                          ALL = FALSE,Cross_val=FALSE){
  output <- list()

  if(Cross_val){
    K_fold = 5
    folds <- createFolds(y, k = K_fold)

    Mat_value <- array(NA,dim=c(length(K_set),length(Q_set),K_fold))
    dimnames(Mat_value) <-  list(
      Dim1 = as.character(K_set),
      Dim2 = as.character(Q_set),
      Dim3 = as.character(1:K_fold)
    )

    for(K in K_set){
      for(Q in Q_set){
        for(n_fold in 1:K_fold){
          idx_test = folds[[n_fold]]
          X_test = X[idx_test,]
          y_test = y[idx_test]
          X_train = X[-idx_test,]
          y_train = y[-idx_test]
          model <- MoEBIUS_reg_fixe(X_train,y_train,K,Q,learning_rate,iter_max, init_type)
          Mat_value[as.character(K),as.character(Q),n_fold] <- mean((prediction_MoEBIUS_reg(model,X_test) - y_test)**2)
        }
      }
    }
    Mat_value <- apply(Mat_value,c(1,2),mean)
    idx_min <- arrayInd(which.min(Mat_value), dim(Mat_value))
    model_CV <- MoEBIUS_reg_fixe(X,y,K_set[idx_min[1]],Q_set[idx_min[2]],learning_rate,iter_max, init_type)

    output$model_CV <- model_CV
  }

  if(ALL){
    model_save <- list()
    iter_save = 1
  }

  model_best_elbo <- list()
  model_best_elbo$elbo <- -Inf
  model_best_ICL <- list()
  model_best_ICL$ICL <- -Inf
  for(K in K_set){
    for(Q in Q_set){
      model <- MoEBIUS_reg_fixe(X,y,K,Q,learning_rate,iter_max, init_type)
      if(ALL){model_save[[iter_save]] <- model;  iter_save = iter_save +1}
      if(model_best_elbo$elbo < model$elbo){model_best_elbo <- model}
      if(model_best_ICL$ICL < model$ICL){model_best_ICL <- model}
    }
  }


  output$best_ICL <- model_best_ICL
  output$best_ELBO <- model_best_elbo
  if(ALL){output$save <- model_save}
  return(output)

}

