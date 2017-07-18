"""
Classes for predictors and special layers.
"""
import dynet
import numpy as np

from constants import BALANCED, IMBALANCED


class SequencePredictor:
    """Convenience class to wrap a sequence prediction model."""
    def __init__(self, builder):
        """Initializes the model. Expects a LSTMBuilder or SimpleRNNBuilder."""
        self.builder = builder
    
    def predict_sequence(self, inputs):
        """Predicts the output of a sequence."""
        return [self.builder(x) for x in inputs]


class RNNSequencePredictor(SequencePredictor):
    """Convenience class to wrap an RNN model."""
    def predict_sequence(self, inputs):
        s_init = self.builder.initial_state()
        return [x.output() for x in s_init.add_inputs(inputs)]


class BiRNNSequencePredictor(SequencePredictor):
    """Convenience class to wrap an LSTM builder."""
    def predict_sequence(self, f_inputs, b_inputs):
        f_init = self.builder.initial_state()
        b_init = self.builder.initial_state()
        forward_sequence = [x.output() for x in f_init.add_inputs(f_inputs)]
        backward_sequence = [x.output() for x in b_init.add_inputs(
            reversed(b_inputs))]
        return forward_sequence, backward_sequence


class CrossStitchLayer:
    """Cross-stitch layer class."""
    def __init__(self, model, num_tasks, hidden_dim, num_subspaces=1,
                 init_scheme=BALANCED):
        """
        Initializes a CrossStitchLayer.
        :param model: the DyNet Model
        :param num_tasks: the number of tasks
        :param hidden_dim: the # of hidden dimensions of the previous LSTM layer
        :param num_subspaces: the number of subspaces
        :param init_scheme: the initialization scheme; balanced or imbalanced
        """
        print('Using %d subspaces...' % num_subspaces, flush=True)
        alpha_params = np.full((num_tasks * num_subspaces,
                                num_tasks * num_subspaces),
                               1. / (num_tasks * num_subspaces))
        if init_scheme == IMBALANCED:
            if num_subspaces == 1:
                alpha_params = np.full((num_tasks, num_tasks),
                                       0.1 / (num_tasks - 1))
                for i in range(num_tasks):
                    alpha_params[i, i] = 0.9
            else:
                # 0 1 0 1
                # 0 1 0 1
                # 1 0 1 0
                # 1 0 1 0
                for (x, y), value in np.ndenumerate(alpha_params):
                    if (y + 1) % num_subspaces == 0 and not \
                            (x in range(num_tasks, num_tasks+num_subspaces)):
                        alpha_params[x, y] = 0.95
                    elif (y + num_subspaces) % num_subspaces == 0 and x \
                            in range(num_tasks, num_tasks+num_subspaces):
                        alpha_params[x, y] = 0.95
                    else:
                        alpha_params[x, y] = 0.05

        self.alphas = model.add_parameters(
            (num_tasks*num_subspaces, num_tasks*num_subspaces),
            init=dynet.NumpyInitializer(alpha_params))
        print('Initializing cross-stitch units to:', flush=True)
        print(dynet.parameter(self.alphas).value(), flush=True)
        self.num_tasks = num_tasks
        self.num_subspaces = num_subspaces
        self.hidden_dim = hidden_dim

    def stitch(self, predictions):
        """
        Takes as inputs a list of the predicted states of the previous layers of
        each task, e.g. for two tasks a list containing two lists of
        n-dimensional output states. For every time step, the predictions of
        each previous task layer are then multiplied with the cross-stitch
        units to obtain a linear combination. In the end, we obtain a list of
        lists of linear combinations of states for every subsequent task layer.
        :param predictions: a list of length num_tasks containing the predicted
                            states for each task
        :return: a list of length num_tasks containing the linear combination of
                 predictions for each task
        """
        assert self.num_tasks == len(predictions)
        linear_combinations = []
        # iterate over tuples of predictions of each task at every time step
        for task_predictions in zip(*predictions):
            # concatenate the predicted state for all tasks to a matrix of shape
            # (num_tasks*num_subspaces, hidden_dim/num_subspaces);
            # we can multiply this directly with the alpha values
            concat_task_predictions = dynet.reshape(
                dynet.concatenate_cols(list(task_predictions)),
                (self.num_tasks*self.num_subspaces,
                 self.hidden_dim / self.num_subspaces))

            # multiply the alpha matrix with the concatenated predictions to
            # produce a linear combination of predictions
            alphas = dynet.parameter(self.alphas)
            product = alphas * concat_task_predictions
            if self.num_subspaces != 1:
                product = dynet.reshape(product,
                                        (self.num_tasks, self.hidden_dim))
            linear_combinations.append(product)

        stitched = [linear_combination for linear_combination in
                    zip(*linear_combinations)]
        return stitched


class LayerStitchLayer:
    """Layer-stitch layer class."""
    def __init__(self, model, num_layers, hidden_dim, init_scheme=IMBALANCED):
        """
        Initializes a LayerStitchLayer.
        :param model: the DyNet model
        :param num_layers: the number of layers
        :param hidden_dim: the hidden dimensions of the LSTM layers
        :param init_scheme: the initialisation scheme; balanced or imbalanced
        """
        if init_scheme == IMBALANCED:
            beta_params = np.full((num_layers), 0.1 / (num_layers - 1))
            beta_params[-1] = 0.9
        elif init_scheme == BALANCED:
            beta_params = np.full((num_layers), 1. / num_layers)
        else:
            raise ValueError('Invalid initialization scheme for layer-stitch '
                             'units: %s.' % init_scheme)
        self.betas = model.add_parameters(
            num_layers, init=dynet.NumpyInitializer(beta_params))
        print('Initializing layer-stitch units to:', flush=True)
        print(dynet.parameter(self.betas).value(), flush=True)
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

    def stitch(self, layer_predictions):
        """
        Takes as input the predicted states of all the layers of a task-specific
        network and produces a linear combination of them.
        :param layer_predictions: a list of length num_layers containing lists
                                  of length seq_len of predicted states for
                                  each layer
        :return: a list of linear combinations of the predicted states at every
                time step for each layer
        """
        assert len(layer_predictions) == self.num_layers
        linear_combinations = []
        # iterate over tuples of predictions of each layer at every time step
        for layer_states in zip(*layer_predictions):
            # concatenate the predicted state for all layers to a matrix of
            # shape (num_layers, hidden_dim)
            concatenated_layer_states = dynet.reshape(dynet.concatenate_cols(
                list(layer_states)), (self.num_layers, self.hidden_dim))

            # multiply with (1, num_layers) betas to produce (1, hidden_dim)
            product = dynet.transpose(dynet.parameter(
                self.betas)) * concatenated_layer_states

            # reshape to (hidden_dim)
            reshaped = dynet.reshape(product, (self.hidden_dim,))
            linear_combinations.append(reshaped)
        return linear_combinations


class Layer:
    """Class for a single layer or a two-layer MLP."""
    def __init__(self, model, in_dim, output_dim, activation=dynet.tanh,
                 mlp=False):
        """
        Initialize the layer and add its parameters to the model.
        :param model: the DyNet Model
        :param in_dim: the input dimension
        :param output_dim: the output dimension
        :param activation: the activation function that should be used
        :param mlp: if True, add a hidden layer with 100 dimensions
        """
        self.act = activation
        self.mlp = mlp
        if mlp:
            mlp_dim = 100
            self.W_mlp = model.add_parameters((mlp_dim, in_dim))
            self.b_mlp = model.add_parameters((mlp_dim))
        else:
            mlp_dim = in_dim
        self.W_out = model.add_parameters((output_dim, mlp_dim))
        self.b_out = model.add_parameters((output_dim))
        
    def __call__(self, x):
        if self.mlp:
            W_mlp = dynet.parameter(self.W_mlp)
            b_mlp = dynet.parameter(self.b_mlp)
            input = dynet.rectify(W_mlp*x + b_mlp)
        else:
            input = x
        W_out = dynet.parameter(self.W_out)
        b_out = dynet.parameter(self.b_out)
        act = self.act(W_out*input + b_out)
        return act
