import numpy as np


def make_2d_grid(height=40,width=40):
    """
    makes a grid of X and Y coordinates, returns meshgrid and (width * height)^2 distance matrix  
    Returns: 
        X: vector of X-coordinates (P=width*height)
        Y: vector of Y-ccordinates 
        D: PxP matrix of Eucledian distances 
    """
    x, y = np.meshgrid(np.arange(width), np.arange(height))
    X = x.ravel().astype(float)
    Y = y.ravel().astype(float)
    Distance = np.sqrt((X[:, None] - X[None, :]) ** 2 +
                       (Y[:, None] - Y[None, :]) ** 2)
    return X, Y, Distance


def make_gaussian_field(X,Y,D,K,sigma): 
    """
    returns a Gaussian random field with mean 0, standard deviation of 1 and a smoothness of sigma kernel width
    Args: 
        X: vector of X-coordinates (P=width*height)
        Y: vector of Y-ccordinates 
        D: PxP matrix of Eucledian distances 
        K: dimensionality of random field
        sigma: width of Gaussian smoothness kernel 
    Returns: 
        F (ndarray): K x P array of random numbers of a Gaussian random field
    """
    P = D.shape[0]
    # Gaussian covariance between locations: unit variance, correlation
    # falls off as exp(-d^2 / (2 sigma^2))
    C = np.exp(-D ** 2 / (2 * sigma ** 2))
    # Matrix square root via eigendecomposition (robust for near-singular C)
    w, V = np.linalg.eigh(C)
    A = V * np.sqrt(np.clip(w, 0, None))
    F = np.random.normal(size=(K, P)) @ A.T
    return F