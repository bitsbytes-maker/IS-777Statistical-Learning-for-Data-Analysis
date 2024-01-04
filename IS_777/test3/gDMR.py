
import numpy as np
import scipy.special as special
import scipy.optimize as optimize
import scipy.misc as misc
from glda import LDA

class gDMR(LDA):
    '''
    Topic Model with Dirichlet Multinomial Regression
    '''
    def __init__(self, G, sigma, beta, docs, vecs, V, trained=None):
        super( gDMR, self ).__init__(G, 0.0, beta, docs, V, trained)
        self.L = vecs.shape[1]  # Feature vector length (dimensionality)
        self.vecs = vecs  # Document feature vectors
        self.sigma = sigma # Sigma parameter for the normal distribution
         # Initialize Lambda with a multivariate normal distribution
        self.Lambda = np.random.multivariate_normal(np.zeros(self.L),
            (self.sigma ** 2) * np.identity(self.L), size=self.G)
         # Initialize Gamma with a gamma distribution (shape=1, scale=1)
        self.Gamma = np.random.gamma(1,1, size= self.G)
        # Initialize alpha_i with a gamma distribution (shape=1, scale=1)
        self.alpha_i = np.random.gamma(1, 1, size=self.G)


        # Calculate the alpha parameter
        # This assumes vecs is a 2D array with shape (number of documents, number of features)
        # and that Lambda is a 2D array with shape (number of groups, number of features)
        if trained is None:
            self.alpha_prime = self.get_alpha_prime()
        else:
            # Using parameters from a trained model
            self.Lambda = trained.Lambda
            self.alpha_prime = trained.alpha_prime


    def hyperparameter_learning(self):
        '''
        update alpha (overwrite)
        '''
        if self.trained is None:
            self.bfgs()
            self.alpha_prime = self.get_alpha_prime()
            self.alpha = self.get_alpha()
   
    def get_alpha(self, Lambda=None):
        '''
        alpha = exp(Lambda^T x)
        '''
        if Lambda is None:
            if self.trained is None:
                Lambda = self.Lambda
            else:
                Lambda = self.trained.Lambda
        return np.exp(np.dot(self.vecs, Lambda.T))

    def get_alpha_n_m_z(self, idx=None):
        # Compute alpha_prime using the new formula
        alpha_prime = self.get_alpha_prime()

        if idx is None:
            # If no specific index is provided, return the sum of n_m_z and alpha_prime
            # Ensure that alpha_prime is correctly shaped for broadcasting
            if alpha_prime.shape[1] == 1:
                alpha_prime = alpha_prime.reshape(-1)
            return self.n_m_z + alpha_prime
        else:
            # If a specific index is provided, ensure it is within bounds
            if idx < len(self.n_m_z):
                # Get alpha_prime for the specific index and ensure correct shape
                alpha_prime_idx = alpha_prime[idx]
                if alpha_prime_idx.ndim > 1 and alpha_prime_idx.shape[1] == 1:
                    alpha_prime_idx = alpha_prime_idx.reshape(-1)
                return self.n_m_z[idx] + alpha_prime_idx
            else:
                # Handle out-of-bounds index
                # Depending on your model's requirement, you can return a default value or raise an error
                raise IndexError("Index out of bounds in get_alpha_n_m_z: idx={}".format(idx))

    # def get_alpha_n_m_z(self, idx=None):
    #     # Recalculate alpha using the new formula
    #     if idx is None:
    #         alpha_prime = self.get_alpha_prime()  # Assume get_alpha_prime computes the new alpha
    #         return self.n_m_z + alpha_prime
    #     else:
    #         alpha_prime = self.get_alpha_prime()[idx]  # get_alpha_prime accounts for the idx
    #         return self.n_m_z[idx] + alpha_prime[idx]

    def get_alpha_prime(self, Lambda=None):
        '''
        alpha_prime = alpha_i * exp(Lambda^T x) + gamma
        '''
        if Lambda is None:
            if self.trained is None:
                Lambda = self.Lambda
            else:
                Lambda = self.trained.Lambda
                
        # Calculate the exponentiated dot product
        exp_dot_product = np.exp(np.dot(self.vecs, Lambda.T))
        product = self.alpha_i * exp_dot_product
        # Multiply by alpha_i and add Gamma for each document
        # print(self.alpha_i.shape)
        # print(exp_dot_product.shape)
        # print(self.Gamma.shape)
        # print(Lambda.shape)
        # print(product.shape)
        alpha_prime = product + self.Gamma
        return alpha_prime

    def bfgs(self):
        def ll(x):
            x = x.reshape((self.G, self.L))
            # print("Shape of x in _ll:", x.shape)
            return self._ll(x)

        def dll(x):
            x = x.reshape((self.G, self.L))
            # print("Shape of x in _dll:", x.shape)
            result = self._dll(x)
            result = result.reshape(self.G * self.L)
            return result

        Lambda = np.random.multivariate_normal(np.zeros(self.L), 
            (self.sigma ** 2) * np.identity(self.L), size=self.G)
        Lambda = Lambda.reshape(self.G * self.L)
        # print("Shape of Lambda before optimization:", Lambda.shape)

        newLambda, fmin, res = optimize.fmin_l_bfgs_b(ll, Lambda, dll)
        self.Lambda = newLambda.reshape((self.G, self.L))
        # print("Shape of Lambda after optimization:", self.Lambda.shape)

    def _ll(self, x):
        # print("Shape of x in -_ll:", x.shape)
        result = 0.0
        # P(w|z)
        result += self.G * special.gammaln(self.beta * self.G)
        result += -np.sum(special.gammaln(np.sum(self.n_z_w, axis=1)))
        result += np.sum(special.gammaln(self.n_z_w))
        result += -self.V * special.gammaln(self.beta)

        # P(z|Lambda)
        alpha_prime = self.get_alpha_prime(x)
        result += np.sum(special.gammaln(np.sum(alpha_prime, axis=1)))
        result += -np.sum(special.gammaln(
            np.sum(self.n_m_z + alpha_prime, axis=1)))
        result += np.sum(special.gammaln(self.n_m_z + alpha_prime))
        result += -np.sum(special.gammaln(alpha_prime))

        # P(Lambda)
        result += -self.G / 2.0 * np.log(2.0 * np.pi * (self.sigma ** 2))
        result += -1.0 / (2.0 * (self.sigma ** 2)) * np.sum(x ** 2)

        result = -result
        return result

    def _dll(self, x):
        alpha_prime = self.get_alpha_prime(x)
        alpha = self.get_alpha(x)
        result = np.sum(self.alpha_i[np.newaxis, :, np.newaxis] * self.vecs[:,np.newaxis,:] * alpha[:,:,np.newaxis]\
            * (special.digamma(np.sum(alpha_prime, axis=1))[:,np.newaxis,np.newaxis]\
            - special.digamma(np.sum(self.n_m_z + alpha_prime, axis=1))[:,np.newaxis,np.newaxis]\
            + special.digamma(self.n_m_z + alpha_prime)[:,:,np.newaxis]\
            - special.digamma(alpha_prime)[:,:,np.newaxis]), axis=0)\
            - x / (self.sigma ** 2)
        result = -result
        return result

    def params(self):
        return '''G=%d, sigma=%s, beta=%s''' % (self.G, self.sigma, self.beta)