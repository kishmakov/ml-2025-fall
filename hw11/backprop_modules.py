import numpy as np


class Module(object):
    """
    Basically, you can think of a module as of a something (black box)
    which can process `input` data and produce `ouput` data.
    This is like applying a function which is called `forward`:

        output = module.forward(input)

    The module should be able to perform a backward pass: to differentiate the `forward` function.
    Moreover, it should be able to differentiate it if is a part of chain (chain rule).
    The latter implies there is a gradient from previous step of a chain rule.

        input_grad = module.backward(input, output_grad)
    """
    def __init__ (self):
        self._output = None
        self._input_grad = None
        self.training = True

    def forward(self, input):
        """
        Takes an input object, and computes the corresponding output of the module.
        """
        self._output = self._compute_output(input)
        return self._output

    def backward(self, input, output_grad):
        """
        Performs a backpropagation step through the module, with respect to the given input.

        This includes
         - computing a gradient w.r.t. `input` (is needed for further backprop),
         - computing a gradient w.r.t. parameters (to update parameters while optimizing).
        """
        self._input_grad = self._compute_input_grad(input, output_grad)
        self._update_parameters_grad(input, output_grad)
        return self._input_grad


    def _compute_output(self, input):
        """
        Computes the output using the current parameter set of the class and input.
        This function returns the result which will be stored in the `_output` field.

        Example: in case of identity operation:

        output = input
        return output
        """
        raise NotImplementedError


    def _compute_input_grad(self, input, output_grad):
        """
        Returns the gradient of the module with respect to its own input.
        The shape of the returned value is always the same as the shape of `input`.

        Example: in case of identity operation:
        input_grad = output_grad
        return input_grad
        """

        raise NotImplementedError

    def _update_parameters_grad(self, input, output_grad):
        """
        Computing the gradient of the module with respect to its own parameters.
        No need to override if module has no parameters (e.g. ReLU).
        """
        pass

    def zero_grad(self):
        """
        Zeroes `gradParams` variable if the module has params.
        """
        pass

    def get_parameters(self):
        """
        Returns a list with its parameters.
        If the module does not have parameters return empty list.
        """
        return []

    def get_parameters_grad(self):
        """
        Returns a list with gradients with respect to its parameters.
        If the module does not have parameters return empty list.
        """
        return []

    def train(self):
        """
        Sets training mode for the module.
        Training and testing behaviour differs for Dropout, BatchNorm.
        """
        self.training = True

    def evaluate(self):
        """
        Sets evaluation mode for the module.
        Training and testing behaviour differs for Dropout, BatchNorm.
        """
        self.training = False

    def __repr__(self):
        """
        Pretty printing. Should be overrided in every module if you want
        to have readable description.
        """
        return "Module"


class BatchNormalization(Module):
    EPS = 1e-3

    def __init__(self, alpha=0.):
        super(BatchNormalization, self).__init__()
        self.alpha = alpha
        self.moving_mean = 0.
        self.moving_variance = 1.

    def _compute_output(self, input):
        # Training: use batch statistics and update moving averages
        if self.training:
            batch_mean = np.mean(input, axis=0)
            batch_var = np.var(input, axis=0)
            # Update moving averages
            self.moving_mean = self.moving_mean * self.alpha + batch_mean * (1 - self.alpha)
            self.moving_variance = self.moving_variance * self.alpha + batch_var * (1 - self.alpha)
            output = (input - batch_mean) / np.sqrt(batch_var + self.EPS)
        else:
            output = (input - self.moving_mean) / np.sqrt(self.moving_variance + self.EPS)
        return output

    def _compute_input_grad(self, input, output_grad):
        # Gradient with respect to input
        if self.training:
            N = input.shape[0]
            batch_mean = np.mean(input, axis=0)
            batch_var = np.var(input, axis=0)
            var_eps = batch_var + self.EPS
            std = np.sqrt(var_eps)
            # BN backward (no gamma/beta):
            # dx = (1/std) * (dy - mean(dy) - (x-mean) * mean(dy*(x-mean))/var_eps)
            dy = output_grad
            mean_dy = np.mean(dy, axis=0)
            mean_dy_xmu = np.mean(dy * (input - batch_mean), axis=0)
            grad_input = (dy - mean_dy - (input - batch_mean) * (mean_dy_xmu / var_eps)) / std
        else:
            grad_input = output_grad / np.sqrt(self.moving_variance + self.EPS)
        return grad_input

    def __repr__(self):
        return "BatchNormalization"


class ChannelwiseScaling(Module):
    """
       Implements linear transform of input y = \\gamma * x + \\beta
       where \\gamma, \\beta - learnable vectors of length x.shape[-1]
    """
    def __init__(self, n_out):
        super(ChannelwiseScaling, self).__init__()

        stdv = 1./np.sqrt(n_out)
        self.gamma = np.random.uniform(-stdv, stdv, size=n_out)
        self.beta = np.random.uniform(-stdv, stdv, size=n_out)

        self.gradGamma = np.zeros_like(self.gamma)
        self.gradBeta = np.zeros_like(self.beta)

    def _compute_output(self, input):
        output = input * self.gamma + self.beta
        return output

    def _compute_input_grad(self, input, output_grad):
        grad_input = output_grad * self.gamma
        return grad_input

    def _update_parameters_grad(self, input, output_grad):
        self.gradBeta = np.sum(output_grad, axis=0)
        self.gradGamma = np.sum(output_grad*input, axis=0)

    def zero_grad(self):
        self.gradGamma.fill(0)
        self.gradBeta.fill(0)

    def get_parameters(self):
        return [self.gamma, self.beta]

    def get_parameters_grad(self):
        return [self.gradGamma, self.gradBeta]

    def __repr__(self):
        return "ChannelwiseScaling"



class Dropout(Module):
    def __init__(self, p=0.5):
        super(Dropout, self).__init__()

        self.p = p
        self.mask = []

    def _compute_output(self, input):
        if self.training:
            keep_prob = 1.0 - self.p
            self.mask = (np.random.rand(*input.shape) < keep_prob).astype(input.dtype)
            output = input * self.mask / keep_prob
        else:
            output = input
        return output

    def _compute_input_grad(self, input, output_grad):
        if self.training:
            keep_prob = 1.0 - self.p
            grad_input = output_grad * self.mask / keep_prob
        else:
            grad_input = output_grad
        return grad_input

    def __repr__(self):
        return "Dropout"
