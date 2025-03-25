#' Compute Evidence Lower Bound (ELBO) for CoCoLBM
#'
#' Calculates the Evidence Lower Bound (ELBO) for a Co-Conditional Latent Block Model (CoCoLBM).
#'
#' @param params A list containing model parameters.
#' @return A numeric value representing the ELBO.
#' @export
ELBO_CoCoLBM <- function(params){
  tau <- params$tau
  rho_ks<- params$rho_ks
  nu<- params$nu
  pi_k<- params$pi_k
  tmp <- params$tmp

  N <- dim(tau)[1]
  K <- dim(tmp)[2]
  p <- dim(nu)[2]
  Q <- dim(nu)[3]

  # tmp = array(NA,dim=c(N,K,p,Q))
  # for(k in 1:K){
  #   for(s in 1:Q){
  #     tmp[,k,,s] =dnorm(X_train, mean = mu_ks[k,s], sd = sqrt(sigma2_ks[k,s]), log = TRUE)
  #   }
  # }
  tau_tmp = array(rep(tau, p*Q), dim = c(dim(tau),p,Q))
  nu_tmp = aperm(array(rep(nu, N), dim = c(dim(nu),N)),c(4,1,2,3))

  res = sum(tau_tmp * tmp * nu_tmp)
  res = res + sum(nu * log_err(aperm(array(rep(rho_ks, p),dim=c(K,Q,p)),c(1,3,2))))
  res = res + sum(tau*log_err(matrix(pi_k,N,K,byrow = T)))
  entropie_nu = -sum(nu * log_err(nu))
  entropie_tau = -sum(tau * log_err(tau))

  return(res + entropie_nu + entropie_tau )
}

#' Update Tau for CoCoLBM
#'
#' Updates the tau matrix based on current model parameters.
#'
#' @param params A list containing model parameters.
#' @return A list with updated tau matrix.
#' @export
update_tau_CoCoLBM <- function(params){
  nu<- params$nu
  pi_k<- params$pi_k
  tmp <- params$tmp
  N <- dim(tmp)[1]
  K <- dim(tmp)[2]
  p <- dim(nu)[2]
  Q <- dim(nu)[3]
  # tmp = array(NA,dim=c(N,K,p,Q))
  # for(k in 1:K){
  #   for(s in 1:Q){
  #     tmp[,k,,s] =dnorm(X_train, mean = mu_ks[k,s], sd = sqrt(sigma2_ks[k,s]), log = TRUE)
  #   }
  # }
  nu_tmp = aperm(array(rep(nu, N), dim = c(dim(nu),N)),c(4,1,2,3))
  tmp2 = tmp * nu_tmp
  tmp2 = apply(tmp2,c(1,2),sum)
  Tik = tmp2 + matrix(log(pi_k),N,K,byrow = T) - 1
  res = Softmax_log(Tik)

  params$tau <- res
  return(params)
}

#' Update Nu for CoCoLBM
#'
#' Updates the nu tensor for Co-Conditional Latent Block Model (CoCoLBM).
#'
#' @param params A list containing model parameters.
#' @return A list with updated nu matrix.
#' @export
update_nu_CoCoLBM <- function(params){
  tau <- params$tau
  rho_ks<- params$rho_ks
  tmp <- params$tmp

  N <- dim(tau)[1]
  K <- dim(tau)[2]
  p <- dim(tmp)[3]
  Q <- dim(rho_ks)[2]
  # tmp = array(NA,dim=c(N,K,p,Q))
  # for(k in 1:K){
  #   for(s in 1:Q){
  #     tmp[,k,,s] =dnorm(X_train, mean = mu_ks[k,s], sd = sqrt(sigma2_ks[k,s]), log = TRUE)
  #   }
  # }
  tau_tmp = array(rep(tau, p*Q), dim = c(dim(tau),p,Q))
  tmp2 = tmp * tau_tmp
  tmp2 = apply(tmp2,c(2,3,4),sum)
  V_kjs = tmp2 + aperm(array(rep(log(rho_ks), p),dim=c(K,Q,p)),c(1,3,2)) - 1 #remet en dim KxpxQ
  res = aperm(array(sapply(1:K,function(k){Softmax_log(V_kjs[k,,])}),dim=c(p,Q,K)),c(3,1,2))

  params$nu <- res
  return(params)
}

#' Update Pi for CoCoLBM
#'
#' Updates the pi_k vector, representing the proportion of individuals in each latent class.
#'
#' @param params A list containing model parameters.
#' @return A list with updated pi_k vector.
#' @export
update_pi_CoCoLBM <- function(params){
  tau <- params$tau
  N <- dim(tau)[1]
  params$pi_k <- colSums(tau)/N
  return(params)
}

#' Update Rho for CoCoLBM
#'
#' Updates the rho_ks matrix, which represents the latent probabilities for variables.
#'
#' @param params A list containing model parameters.
#' @return A list with updated rho_ks matrix.
#' @export
update_rho_CoCoLBM <- function(params){
  nu <- params$nu
  p <- dim(nu)[2]
  params$rho_ks <- apply(nu,c(1,3),sum)/p
  return(params)
}

#' Update Mu for CoCoLBM
#'
#' Updates the mu_ks matrix, which represents the mean values for each class and component.
#'
#' @param params A list containing model parameters.
#' @return A list with updated mu_ks matrix.
#' @export
update_mu_CoCoLBM <- function(params){
  X_train <- params$X
  tau <- params$tau
  nu <- params$nu

  N <- dim(tau)[1]
  K <- dim(tau)[2]
  p <- dim(nu)[2]
  Q <- dim(nu)[3]
  res = matrix(NA,K,Q)
  for(k in 1:K){
    for(s in 1:Q){
      res[k,s] = (tau[,k] %*% X_train %*% nu[k,,s]) / (tau[,k] %*% matrix(1,N,p) %*% nu[k,,s]) #OK mu_ks
    }
  }

  params$mu_ks <- res

  return(params)
}

#' Update Sigma Squared for CoCoLBM
#'
#' Updates the sigma2_ks matrix, representing the variance for each class and component.
#'
#' @param params A list containing model parameters.
#' @return A list with updated sigma2_ks matrix.
#' @export
update_sigma2_CoCoLBM <- function(params){

  X_train <- params$X
  tau <- params$tau
  nu <- params$nu
  mu_ks <- params$mu_ks

  N <- dim(tau)[1]
  K <- dim(tau)[2]
  p <- dim(nu)[2]
  Q <- dim(nu)[3]
  res = matrix(NA,K,Q)
  for(k in 1:K){
    for(s in 1:Q){
      res[k,s] = (tau[,k] %*% ((X_train - mu_ks[k,s])**2) %*% nu[k,,s]) / (tau[,k] %*% matrix(1,N,p) %*% nu[k,,s]) #OK sigma2
    }
  }


  params$sigma2_ks <- pmax(res,.Machine$double.eps)
  return(params)
}

#' Initialize Parameters for CoCoLBM
#'
#' Initializes the parameters tau and nu for Co-Conditional Latent Block Model (CoCoLBM) using k-means clustering.
#'
#' @param X The data matrix (NxP) for training.
#' @param K Number of latent classes for rows.
#' @param Q Number of latent classes for columns.
#' @return A list containing initialized parameters.
#' @export
InitializationCoCoLBM <- function(X,K,Q){
  params <- list()
  params$X <- X
  params$K <- K
  params$Q <- Q


  N <- dim(X)[1]
  p <- dim(X)[2]

  params$N <- N
  params$p <- p
  tau <- one_hot_errormachine(tryCatch({ kmeans(X,centers=K,nstart = 50,iter.max = 10)$cluster},
                                       error=function(cond){
                                         #message("Kmeans with small noise")
                                         mat2 <- X + matrix(runif(n = nrow(X) * ncol(X),min=0,max=.1e-8),nrow(X_train),ncol(X_train))
                                         km= kmeans(mat2,centers=K,nstart = 50)$cluster
                                         return(km)
                                       }))

  clust_tau = apply(tau,1,which.max)
  nu <- array(NA,dim = c(K,p,Q))
  for(k in 1:K){
    mat = t(X[clust_tau == k,])
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

  return(params)
}

#' Compute Log Probability for CoCoLBM
#'
#' Calculates the log-probability matrix tmp for the model.
#'
#' @param params A list containing model parameters: X, N, K, p, Q, mu_ks, sigma2_ks.
#' @return A list with updated tmp matrix.
#' @export
calcul_log_prob_CoCoLBM <- function(params){

  tmp = array(NA,dim=c(params$N,params$K,params$p,params$Q))
  for(k in 1:params$K){
    for(s in 1:params$Q){
      #tmp[,k,,s] = dpois(X_train, lambda = mu_ks[k,s], log = TRUE)
      tmp[,k,,s] =dnorm(params$X, mean =params$mu_ks[k,s], sd = sqrt(params$sigma2_ks[k,s]), log = TRUE)
    }
  }
  params$tmp <- tmp
  return(params)
}

#' Fixed CoCoLBM Model
#'
#' Fits a Co-Conditional Latent Block Model (CoCoLBM) with fixed K and Q.
#'
#' @param X The data matrix.
#' @param K Number of latent classes for rows.
#' @param Q Number of latent classes for columns.
#' @param iter_max Maximum number of iterations.
#' @param init_type Initialization method.
#' @return A list containing the fitted model parameters and metrics.
#' @export
CocoLBM_fixe <- function(X,K,Q,iter_max=10,init_type='Kmeans'){
  params_CoCoLBM <- InitializationCoCoLBM(simus$X,K,Q)
  params_CoCoLBM$iter_max = iter_max
  n_iter=1
  while(n_iter <= iter_max){
    params_CoCoLBM <- update_pi_CoCoLBM(params_CoCoLBM)
    params_CoCoLBM <- update_rho_CoCoLBM(params_CoCoLBM)
    params_CoCoLBM <- update_mu_CoCoLBM(params_CoCoLBM)
    params_CoCoLBM <- update_sigma2_CoCoLBM(params_CoCoLBM)

    params_CoCoLBM <- calcul_log_prob_CoCoLBM(params_CoCoLBM)

    params_CoCoLBM <- update_nu_CoCoLBM(params_CoCoLBM)
    params_CoCoLBM <- update_tau_CoCoLBM(params_CoCoLBM)
    #print(ELBO_CoCoLBM(params))


    n_iter = n_iter+1
  }

  params_CoCoLBM$elbo <- ELBO_CoCoLBM(params_CoCoLBM)

  params_CoCoLBM$ICL <-  params_CoCoLBM$elbo -1/2 * (params_CoCoLBM$K-1)*log(params_CoCoLBM$N) -
    1/2 * params_CoCoLBM$K*(params_CoCoLBM$Q-1)*log(params_CoCoLBM$p) -params_CoCoLBM$K * params_CoCoLBM$Q * log( params_CoCoLBM$N* params_CoCoLBM$p)
  return(params_CoCoLBM)
}

#' Co-CoLBM Model Selection
#'
#' Fits a Co-Conditional Latent Block Model and selects the best model based on ELBO and ICL criteria.
#'
#' @param X The data matrix.
#' @param K_set A set of values for the number of row classes.
#' @param Q_set A set of values for the number of column classes.
#' @param iter_max Maximum number of iterations.
#' @param init_type Initialization method.
#' @param ALL Boolean, whether to save all models evaluated.
#' @return A list containing the best models based on ICL and ELBO, and optionally all models.
#' @export
CocoLBM <- function(X,K_set,Q_set,iter_max=10, init_type='Kmeans',ALL=FALSE){

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
      model <- CocoLBM_fixe(X,K,Q,iter_max,init_type)
      if(ALL){model_save[[iter_save]] <- model;  iter_save = iter_save +1}
      if(model_best_elbo$elbo < model$elbo){model_best_elbo <- model}
      if(model_best_ICL$ICL < model$ICL){model_best_ICL <- model}
    }
  }

  output <- list()
  output$best_ICL <- model_best_ICL
  output$best_ELBO <- model_best_elbo
  if(ALL){output$save <- model_save}
  return(output)

}
