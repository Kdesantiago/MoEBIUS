#' Supervised Simulation for Conditional Clustered Latent Block Model (CCLBM)
#'
#' Simulates data for a supervised MoEBIUS model, generating latent variables, an observation matrix,
#' and a response vector for supervised analysis.
#'
#' @param N Number of observations (default is `1000`).
#' @param p Number of observed variables (default is `10`).
#' @param K Number of individual classes (default is `2`).
#' @param Q Number of variable groups (default is `3`).
#' @param C Number of classes in the response vector `y` (default is `1`).
#' @param pi_k Vector of probabilities for individual classes (default is uniform).
#' @param rho_ks Probability matrix for variable groups conditioned on individual classes (default is uniform).
#' @param n_informative Number of informative regression parameters (default is `5`).
#' @param Z Latent matrix of individual classes. If `NULL`, generated randomly.
#' @param W Latent matrix of variable groups. If `NULL`, generated randomly.
#' @param beta Regression parameter vector. If `NULL`, generated randomly.
#' @param lambda_ks Matrix of means for simulating observed variables. If `NULL`, generated randomly.
#' @param classif Logical, if `TRUE`, applies a softmax function to `y` for classification (default is `FALSE`).
#'
#' @return A list containing:
#' \item{N, p, K, Q}{Input parameters for the simulation.}
#' \item{Z}{Latent matrix of individual classes.}
#' \item{Z_classif}{Indices of individual classes for each observation.}
#' \item{W}{Latent matrix of variable groups.}
#' \item{W_classif}{Indices of variable groups for each individual.}
#' \item{pi_k}{Probabilities for individual classes.}
#' \item{rho_ks}{Probabilities for variable groups conditioned on individual classes.}
#' \item{param_X}{List containing the mean and variance of the simulated variables.}
#' \item{X}{Simulated observation matrix.}
#' \item{beta}{Regression parameter vector.}
#' \item{y}{Response vector.}
#'
#' @examples
#' \dontrun{
#' data <- simu_CCLBM_supervised(N = 500, p = 8, K = 3, Q = 2, C = 1)
#' }
#' @export
simu_CCLBM_supervised <- function(N = 1000, p = 10,K = 2,Q = 3,C = 1,pi_k = rep(1/K,K), rho_ks = matrix(1/Q,nrow=K,ncol=Q),n_informative = 5,Z = NULL, W=NULL,beta =NULL,lambda_ks=NULL,classif=FALSE){

  # SImulation des variables latentes
  if(is.null(Z)){
    Z <- t(rmultinom(N,size=1,prob=pi_k))
  }

  if(is.null(W)){
    W <- array(t(sapply(1:K,function(k){t(rmultinom(p,size=1,prob=rho_ks[k,]))})),dim=c(K,p,Q))
  }
  W_classif <- apply(W,c(1,2),which.max)
  Z_classif <- apply(Z,1,which.max)
  # Simulation de X
  if(is.null(lambda_ks)){lambda_ks <- matrix(sample(1:10,size = K*Q,replace = T),K,Q)}



  X <- matrix(NA,N,p)
  for(k in 1:K){
    idx_Z = which(Z_classif == k)
    for(s in 1:Q){
      idx_W = which(W_classif[k,] == s)
      #X[idx_Z,idx_W] <- rpois(length(idx_Z)*length(idx_W),lambda_ks[k,s])
      X[idx_Z,idx_W] <-rnorm(length(idx_Z)*length(idx_W),lambda_ks[k,s])
    }
  }

  # Simulation des paramètres du vecteur de régression

  if(is.null(beta)){
    beta = array(0,dim=c(K,Q,C))
    df <- data.frame("K" = rep(c(1:K),each=Q),"Q" = rep(1:Q,K))
    for(c in 1:C){
      mask_informative = t(sample(as.data.frame(t(df)),size = n_informative))
      # Utiliser apply avec <<- pour modifier les valeurs
      apply(mask_informative, 1, function(x) {
        beta[x[1], x[2],c] <<- rnorm(1)#rpois(1,lambda = 1)
      })
    }
  }


  y = array(NA,dim=c(N,C))
  for(k in 1:K){
    idx_Z = which(Z_classif == k)
    y[idx_Z,] = X[idx_Z,] %*% W[k,,] %*% beta[k,,]
  }
  if(classif) {
    y <- Softmax_log(y)
  }
  output <- list()

  output$N <- N
  output$p <- p
  output$K <- K
  output$Q <- Q
  output$Z <- Z
  output$Z_classif <- Z_classif
  output$W <- W
  output$W_classif <- W_classif
  output$pi_k <- pi_k
  output$rho_ks <- rho_ks
  output$param_X <- list(mu_ks = lambda_ks, sigma_ks = matrix(1,K,Q))
  output$X <- X
  output$beta <- beta
  output$y <- y


  return(output)
}
