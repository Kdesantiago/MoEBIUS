#' One Hot Encoding with Error machine
#'
#' @param Z a vector of size N, where Z\[i\] value indicate the cluster membership of observation i.
#' @param size optional parameter, indicating the number of classes (avoid some empty class problems).
#'
#' @return Z a matrix N x K One-Hot-Encoded by rows, where K is the number of clusters.
#' @export
#' @examples
#' Z <- sample(1:4,10,replace=TRUE)
#' Z_OHE <- one_hot_errormachine(Z)
#' print(Z_OHE)
one_hot_errormachine <- function(Z,size=NULL){
  #------------ Objectif ------------
  # Prend en vecteur de classfication en entrée pour le One-Hot-Encode
  # en prenant en compte l'erreur machine possible.

  #------------ Variables ------------
  n <- length(Z)
  if(is.null(size)) K <- length(unique(Z)) else K = size

  #------------ One Hot Encoding + erreur machine ------------
  mat <- matrix(.Machine$double.xmin,n,K)
  mat[cbind(1:n,Z)] <- 1-.Machine$double.xmin

  #------------ Output ------------
  return(mat)
}


#' Softmax for Logit
#'
#' @param log_X a matrix of logits
#'
#' @return Softmax applied on rows
#' @export
#'
#' @examples
Softmax_log <- function(log_X){
  if(!is.matrix(log_X)){log_X <- as.matrix(log_X)}
  K <- ncol(log_X)

  log_X <- log_X - apply(log_X,1,max)

  ## Now going back to exponential with the same normalization
  X <- exp(log_X) #(matrix(1,n,1) %*% pi) * exp(logX)
  X <- pmin(X,.Machine$double.xmax)
  X <- pmax(X,.Machine$double.xmin)
  X <- X / (rowSums(X) %*% matrix(1,1,K))
  X <- pmin(X,1-.Machine$double.xmin)
  X <- pmax(X,.Machine$double.xmin)

  return(X)
}


#' Natural logarithm with tolerance
#'
#' This function computes the natural logarithm of an array of numbers,
#' adding a small tolerance value to avoid taking the logarithm of zero
#' or negative numbers.
#'
#' @param x The input values for which the natural logarithm
#'          will be computed. It should be a numeric vector or array.
#' @param err a numeric value representing the tolerance to be added to each
#'            element of `x`. Default is the machine's double precision epsilon
#'            (i.e., `.Machine$double.eps`).
#'
#' @return A numeric array of the same shape as `x`, containing the natural logarithm
#'         of each element of `x` after adding the specified tolerance `err`.
#'
#' @export
#'
#' @examples
#' log_err(c(1, 2, 3))          # Computes log(1), log(2), log(3)
#' log_err(c(0, 1, 2), err=1e-10) # Computes log(1e-10), log(1), log(2)
log_err <-function(x,err=.Machine$double.eps){
  return(log(x+err))
}
