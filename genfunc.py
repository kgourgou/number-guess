import numpy as np


class level:
    """
    A level of the network, which takes in some input X and returns some output Y.
    """
    def __init__(self, in_dim, out_dim, learning_rate = 1e-3, parent=None, child=None):
        self.W = 0.001 * np.random.randn(out_dim, in_dim)
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.link(parent, child)
        self.learning_rate = learning_rate

    def removeParent(self):
        if self.parent != None:
            self.parent.child = None
            self.parent = None

    def removeChild(self):
        if self.child != None:
            self.child.parent = None
            self.child = None

    def link(self, parent=None, child=None):
        self.parent = parent
        if parent != None:
            parent.removeChild()
            parent.child = self
        self.child = child
        if child != None:
            child.removeParent()
            child.parent = self

    def value(self, X):
        """
        Return Y
        subclasses will override
        """
        self.X = X
        self.Y = self.W.dot(self.X)
        return self.W.dot(self.X)

    def forward(self, X):
        if self.child == None:
            return self.value(X)
        else:
            return self.child.forward(self.value(X))

    def backward(self, dZ_dY):
        """
        dZ_dY (dZ/dY) should be the incoming derivatives of the ultimate output of the network, Z, with respect to output from this level, Y. We use the chain rule to compute dZ/dX, and kick it up to the parent level. We also calculate dZ/dW, and use it to update the parameters.
        """
        try:
            dZ_dY = dZ_dY.reshape(self.Y.shape)
            if self.parent != None:
                self.parent.backward((self.W.T).dot(dZ_dY))
            dZ_dW = dZ_dY.dot(self.X.T)
            self.W -= self.learning_rate * dZ_dW
        except:
            import pdb; pdb.set_trace()

class powers(level):
    def __init__(self, parent=None, child=None):
        self.link(parent, child)

    def value(self, X):
        self.X = X
        self.Y = power_naive(X)
        return self.Y.copy()

    def backward(self, dZ_dY):
        if self.parent != None:
            #TODO: This is one is harder to compute
            pass
        pass

class const_lin_trans(level):
    def __init__(self, L, parent=None, child=None):
        self.L = L # Linear Transformation
        self.link(parent, child)

    def value(self, X):
        self.X = X
        self.Y = self.L.dot(X)
        return self.Y.copy()

    def grad(self):
        return self.L.copy()

    def backward(self, dZ_dY):
        if self.parent != None:
            self.parent.backward(self.grad().dot(dZ_dY))



class sqeuclidean_loss(level):
    def __init__(self, target, parent=None):
        self.target = target
        self.link(parent)

    def value(self, X):
        self.X = X
        self.Y = ((X - self.target)**2).sum()
        return self.Y

    def grad(self):
        return 2*(self.X-self.target)

    def backward(self):
        if self.parent != None:
            self.parent.backward(self.grad())

class network:
    def __init__(self, first, last):
        self.first = first
        self.last = last

    def input_val(self, X):
        self.X = X

    def grad_desc(self, num_iters=1000, verbose=False):
        loss_history = []

        for it in range(num_iters):
            loss = self.first.forward(self.X)
            loss_history.append(loss)
            self.last.backward()

            if verbose and it % 100 == 0:
                print('iteration {} / {}: loss {}'.format(it, num_iters, loss))

        return loss_history



#class genfunc:
#    def __init__(self, sequence):
#        """
#        A generating function object.
#
#        Input:
#         - seq: A (N,) np.array representing the sequence
#        """
#        self.sequence = sequence
#        self.N = sequence.shape[0]
#
#    def grad_desc(self, learning_rate=1e-3, num_iters=100, verbose=False):
#        # Initialize some weights
#        self.W1 = 0.001 * np.random.randn(1, self.F.shape[0])
#        self.W2 = 0.001 * np.random.randn(1, self.F.shape[0])
#        loss_history = []
#
#        for it in range(num_iters):
#            loss, grad = self.loss()
#            loss_history.append(loss)
#            self.W1 -= learning_rate * grad['W1']
#            self.W2 -= learning_rate * grad['W2']
#
#            if verbose and it % 100 == 0:
#                print('iteration {} / {}: loss {}'.format(it, num_iters, loss))
#
#        return loss_history
#
#    def loss(self):
#        W1 = self.W1
#        W2 = self.W2
#        F = self.F
#        loss = ((W2.dot(F.dot(power_naive(W1.dot(F))) - self.sequence)**2).sum()
#        grad = 2 * (W1.dot(F) - self.sequence).dot(self.F.T)
#        return loss, grad
#
#    def functions(self, func=None):
#        if func == None:
#            func = np.array([
#                [1,0,0,0,0,0,0,0], # 1
#                [0,1,0,0,0,0,0,0], # x
#                [0,0,1,0,0,0,0,0], # x^2
#                [1,1,1,1,1,1,1,1], # 1/(1-x)
#                ])
#        self.F = func

def power_naive(w):
    """
    Given a row vector w representing the first N coefficients of a power series, return a (N,N) matrix where the jth row is the representation of the jth power of w.

    Input:
     - w: an (N,) dimensional np.ndarray
    """
    N = w.shape[0]
    out = np.zeros((N,2*N-1))
    out[0,0] = 1 # setup the first row
    for j in range(1,N):
        out[j] = mult_row_naive(out[j-1], w)[:2*N-1]
    return out


def mult_row_naive(a, b):
    M = a.shape[0]
    N = b.shape[0]
    out = np.zeros(M+N-1)
    for j in range(0,N+M-1):
        for n in range(max(0,j-M+1), min(j+1, N)):
            out[j] += b[n]*a[j-n]
    return out
