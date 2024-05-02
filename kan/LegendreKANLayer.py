import torch



class NaiveLegendreKANLayer(torch.nn.Module):
    def __init__(self, inputdim, outdim, degree, addbias=True):
        super(NaiveLegendreKANLayer, self).__init__()
        self.degree = degree
        self.addbias = addbias
        self.inputdim = inputdim
        self.outdim = outdim

        self.legendrecoeffs = torch.nn.Parameter(torch.randn(outdim, inputdim, degree) / 
                                              (torch.sqrt(inputdim) * torch.sqrt(degree)))
        if self.addbias:
            self.bias = torch.nn.Parameter(torch.zeros(1, outdim))

    def forward(self, x):
        xshp = x.shape
        outshape = xshp[0:-1] + (self.outdim,)
        x = torch.reshape(x, (-1, self.inputdim))

        # Normalize input to [-1, 1]
        x = 2 * (x - x.min()) / (x.max() - x.min()) - 1

        # Compute Legendre polynomials up to the specified degree
        P = [self.legendre(d, x) for d in range(self.degree)]
        P = torch.stack(P, -1)

        # Compute the interpolation of the various functions defined by their Legendre coefficients for each input coordinates and we sum them 
        y = torch.sum(P * self.legendrecoeffs, -1)
        if self.addbias:
            y += self.bias

        y = torch.reshape(y, outshape)
        return y

    def legendre(self, n, x):
        if n == 0:
            return x.new_ones(*x.shape)
        elif n == 1:
            return x
        else:
            return ((2.0 * n - 1.0) * x * self.legendre(n - 1, x) - (n - 1.0) * self.legendre(n - 2, x)) / n