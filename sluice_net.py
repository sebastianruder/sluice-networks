#!/usr/bin/env python3
# coding=utf-8
"""
Sluice Network model.
"""
import random
import os
import numpy as np
import pickle
import dynet
from progress.bar import Bar

from predictors import SequencePredictor, Layer, RNNSequencePredictor, \
    BiRNNSequencePredictor, CrossStitchLayer, LayerStitchLayer
from utils import load_embeddings_file, get_data
from constants import POS, CHUNK, NER, SRL, MODEL_FILE, PARAMS_FILE, \
    IMBALANCED, BALANCED, STITCH, CONCAT, SKIP, NONE, SGD, ADAM


def load(params_file, model_file, args):
    """
    Loads a model by first initializing a model with the hyperparameters
    and then loading the weights of the saved model.
    :param params_file: the file containing the hyperparameters
    :param model_file: the file containing the weights of the saved model
    :return the loaded AdaptNN model
    """
    params = pickle.load(open(params_file, 'rb'))
    model = SluiceNetwork(params['in_dim'],
                          params['h_dim'],
                          params['c_in_dim'],
                          params['h_layers'],
                          params['pred_layer'],
                          params['model_dir'],
                          activation=params['activation'],
                          task_names=params['task_names'],
                          cross_stitch=args.cross_stitch,
                          layer_connect=args.layer_connect,
                          num_subspaces=args.num_subspaces,
                          constraint_weight=args.constraint_weight)
    model.set_indices(params['w2i'], params['c2i'], params['task2tag2idx'])
    model.predictors, model.char_rnn, model.wembeds, model.cembeds = \
        model.build_computation_graph(params['num_words'], params['num_chars'])
    model.model.load(model_file)
    print('Model loaded from %s...' % model_file, flush=True)
    return model


class SluiceNetwork(object):
    def __init__(self, in_dim, h_dim, c_in_dim, h_layers, pred_layer, model_dir,
                 embeds_file=None, activation=dynet.tanh, lower=False,
                 noise_sigma=0.1, task_names=[], cross_stitch=False,
                 layer_connect=NONE, num_subspaces=1, constraint_weight=0,
                 constrain_matrices=[1, 2], cross_stitch_init_scheme=IMBALANCED,
                 layer_stitch_init_scheme=BALANCED):
        """
        :param in_dim: The dimension of the word embeddings.
        :param h_dim: The hidden dimension of the model.
        :param c_in_dim: The dimension of the character embeddings.
        :param h_layers: The number of hidden layers.
        :param pred_layer: Indices indicating at which layer to predict each
                           task, e.g. [1, 2] indicates 1st task is predicted
                           at 1st layer, 2nd task is predicted at 2nd layer
        :param model_dir: The directory where the model should be saved
        :param embeds_file: the file containing pre-trained word embeddings
        :param activation: the DyNet activation that should be used
        :param lower: whether the words should be lower-cased
        :param noise_sigma: the stddev of the Gaussian noise that should be used
                            during training if > 0.0
        :param task_names: the names of the tasks
        :param cross_stitch: whether to use cross-stitch units
        :param layer_connect: the layer connections that are used (stitch,
                              skip, concat, or none)
        :param num_subspaces: the number of subspaces to use (1 or 2)
        :param constraint_weight: weight of subspace orthogonality constraint
                                  (default: 0 = no constraint)
        :param constrain_matrices: indices of LSTM weight matrices that should
                                   be constrained (default: [1, 2])
        :param cross_stitch_init_scheme: initialisation scheme for cross-stitch
        :param layer_stitch_init_scheme: initialisation scheme for layer-stitch
        """
        self.word2id = {}  # word to index mapping
        self.char2id = {}  # char to index mapping
        self.task_names = task_names
        self.main_task = self.task_names[0]
        print('Using the first task as main task:', self.main_task, flush=True)
        self.model_dir = model_dir
        self.model_file = os.path.join(model_dir, MODEL_FILE)
        self.params_file = os.path.join(model_dir, PARAMS_FILE)
        self.cross_stitch = cross_stitch
        self.layer_connect = layer_connect
        self.num_subspaces = num_subspaces
        self.constraint_weight = constraint_weight
        self.constrain_matrices = constrain_matrices
        self.cross_stitch_init_scheme = cross_stitch_init_scheme
        self.layer_stitch_init_scheme = layer_stitch_init_scheme
        self.model = dynet.Model()  # init model
        # term to capture sum of constraints over all subspaces
        self.subspace_penalty = self.model.add_parameters(
            1, init=dynet.NumpyInitializer(np.zeros(1)))
        # weight of subspace constraint
        self.constraint_weight_param = self.model.add_parameters(
            1, init=dynet.NumpyInitializer(np.array(self.constraint_weight)))

        self.task2tag2idx = {}  # need one dictionary per task
        self.pred_layer = pred_layer
        self.in_dim = in_dim
        self.h_dim = h_dim
        self.c_in_dim = c_in_dim
        self.activation = activation
        self.lower = lower
        self.noise_sigma = noise_sigma
        self.h_layers = h_layers
        # keep track of the inner layers and the task predictors
        self.predictors = {'inner': [], 'output_layers_dict': {},
                           'task_expected_at': {}}
        self.wembeds = None  # lookup: embeddings for words
        self.cembeds = None  # lookup: embeddings for characters
        self.embeds_file = embeds_file
        self.char_rnn = None  # RNN for character input

    def save(self):
        """Save model. DyNet only saves parameters. Save rest separately."""
        self.model.save(self.model_file)
        myparams = {"num_words": len(self.word2id),
                    "num_chars": len(self.char2id),
                    "task_names": self.task_names,
                    "w2i": self.word2id,
                    "c2i": self.char2id,
                    "task2tag2idx": self.task2tag2idx,
                    "activation": self.activation,
                    "in_dim": self.in_dim,
                    "h_dim": self.h_dim,
                    "c_in_dim": self.c_in_dim,
                    "h_layers": self.h_layers,
                    "embeds_file": self.embeds_file,
                    "pred_layer": self.pred_layer,
                    'model_dir': self.model_dir,
                    'cross-stitch': self.cross_stitch,
                    'layer-connect': self.layer_connect,
                    'num-subspaces': self.num_subspaces,
                    'constraint-weight': self.constraint_weight,
                    'cross_stitch_init_scheme': self.cross_stitch_init_scheme,
                    'layer_stitch_init_scheme': self.layer_stitch_init_scheme}
        pickle.dump(myparams, open(self.params_file, "wb"))

    def set_indices(self, w2i, c2i, task2t2i):
        """Sets indices of word, character, and task mappings."""
        for task_id in task2t2i:
            self.task2tag2idx[task_id] = task2t2i[task_id]
        self.word2id = w2i
        self.char2id = c2i

    def build_computation_graph(self, num_words, num_chars):
        """Builds the computation graph."""
        # initialize the word embeddings
        if self.embeds_file:
            print('Loading embeddings', flush=True)
            embeddings, emb_dim = load_embeddings_file(self.embeds_file,
                                                       lower=self.lower)
            assert (emb_dim == self.in_dim)
            # initialize all words with embeddings; for very large vocabularies,
            # we don't want to do this
            num_words = len(set(embeddings.keys()).union(set(self.word2id.keys())))
            # init model parameters and initialize them
            wembeds = self.model.add_lookup_parameters((num_words, self.in_dim))
            cembeds = self.model.add_lookup_parameters((num_chars, self.c_in_dim))

            for i, word in enumerate(embeddings.keys()):
                if word not in self.word2id:
                    self.word2id[word] = len(self.word2id.keys())
                wembeds.init_row(self.word2id[word], embeddings[word])
            print('Initialized %d word embeddings...' % i, flush=True)
        else:
            wembeds = self.model.add_lookup_parameters((num_words, self.in_dim))
            cembeds = self.model.add_lookup_parameters((num_chars, self.c_in_dim))

        layers = []  # inner layers
        output_layers_dict = {}  # from task_name to actual softmax predictor
        task_expected_at = {}  # maps task_name => output_layer id

        # connect output layers to tasks
        for output_layer_id, task_name in zip(self.pred_layer, self.task_names):
            assert output_layer_id <= self.h_layers,\
                ('Error: Task cannot be predicted at layer beyond model. '
                 'Increase h_layers.')
            task_expected_at[task_name] = output_layer_id

        print('Task expected at', task_expected_at, flush=True)
        print('h_layers:', self.h_layers, flush=True)

        # we have a separate layer for each task for cross-stitching;
        # otherwise just 1 layer for all tasks with hard parameter sharing
        num_task_layers = len(self.task_names) if self.cross_stitch else 1
        cross_stitch_layers = []
        for layer_num in range(self.h_layers):
            print(">>> %d layer_num" % layer_num, flush=True)
            input_dim = self.in_dim + self.c_in_dim * 2 if layer_num == 0 \
                else self.h_dim
            task_layers = []
            # get one layer per task for cross-stitching or just one layer
            for task_id in range(num_task_layers):
                builder = dynet.LSTMBuilder(1, input_dim, self.h_dim, self.model)
                task_layers.append(BiRNNSequencePredictor(builder))
            layers.append(task_layers)
            if self.cross_stitch:
                print('Using cross-stitch units after layer %d...' % layer_num,
                      flush=True)
                cross_stitch_layers.append(
                    CrossStitchLayer(self.model, len(self.task_names),
                                     self.h_dim, self.num_subspaces,
                                     self.cross_stitch_init_scheme))

        layer_stitch_layers = []

        # store at which layer to predict task
        for task_name in self.task_names:
            task_num_labels = len(self.task2tag2idx[task_name])

            # use a small MLP both for the task losses
            print('Using an MLP for task losses.', flush=True)
            # if we concatenate, the FC layer has to have a larger input_dim
            input_dim = self.h_dim * 2 * self.h_layers\
                if self.layer_connect == CONCAT else self.h_dim * 2
            layer_output = Layer(self.model, input_dim, task_num_labels,
                                 dynet.softmax, mlp=True)
            sequence_predictor = SequencePredictor(layer_output)
            output_layers_dict[task_name] = sequence_predictor

            if self.layer_connect == STITCH:
                print('Using layer-stitch units for task %s...' % task_name,
                      flush=True)
                # w/o cross-stitching, we only use one LayerStitchLayer
                layer_stitch_layers.append(
                    LayerStitchLayer(self.model, self.h_layers, self.h_dim,
                                     self.layer_stitch_init_scheme))

        print('#\nOutput layers: %d\n' % len(output_layers_dict), flush=True)

        # initialize the char RNN
        char_rnn = RNNSequencePredictor(dynet.LSTMBuilder(1, self.c_in_dim, self.c_in_dim, self.model))

        predictors = dict()
        predictors["inner"] = layers
        predictors['cross_stitch'] = cross_stitch_layers
        predictors['layer_stitch'] = layer_stitch_layers
        predictors["output_layers_dict"] = output_layers_dict
        predictors["task_expected_at"] = task_expected_at
        return predictors, char_rnn, wembeds, cembeds

    def fit(self, train_domain, num_epochs, patience, optimizer, train_dir,
            dev_dir):
        """
        Trains the model.
        :param train_domain: the domain used for training
        :param num_epochs: the max number of epochs the model should be trained
        :param patience: the patience to use for early stopping
        :param optimizer: the optimizer that should be used
        :param train_dir: the directory containing the training files
        :param dev_dir: the directory containing the development files
        """
        print("Reading training data from %s..." % train_dir, flush=True)
        train_X, train_Y, _, _, word2id, char2id, task2t2i = get_data(
            [train_domain], self.task_names, data_dir=train_dir, train=True)

        # get the development data of the same domain
        dev_X, dev_Y, org_X, org_Y, _, _, _ = get_data(
            [train_domain], self.task_names, word2id, char2id, task2t2i,
            data_dir=dev_dir, train=False)
        print('Length of training data:', len(train_X), flush=True)
        print('Length of validation data:', len(dev_X), flush=True)

        # store mappings of words and tags to indices
        self.set_indices(word2id, char2id, task2t2i)
        num_words = len(self.word2id)
        num_chars = len(self.char2id)

        print('Building the computation graph...', flush=True)
        self.predictors, self.char_rnn, self.wembeds, self.cembeds = \
            self.build_computation_graph(num_words, num_chars)

        if optimizer == SGD:
            trainer = dynet.SimpleSGDTrainer(self.model)
        elif optimizer == ADAM:
            trainer = dynet.AdamTrainer(self.model)
        else:
            raise ValueError('%s is not a valid optimizer.' % optimizer)

        train_data = list(zip(train_X, train_Y))

        num_iterations = 0
        num_epochs_no_improvement = 0
        best_dev_acc = 0

        print('Training model with %s for %d epochs and patience of %d.'
              % (optimizer, num_epochs, patience))
        for epoch in range(num_epochs):
            print('', flush=True)
            bar = Bar('Training epoch %d/%d...' % (epoch+1, num_epochs),
                      max=len(train_data), flush=True)

            # keep track of the # of updates, total loss, and total # of
            # predicted instances per task
            task2num_updates = {task: 0 for task in self.task_names}
            task2total_loss = {task: 0.0 for task in self.task_names}
            task2total_predicted = {task: 0.0 for task in self.task_names}
            total_loss = 0.0
            total_penalty = 0.0
            total_predicted = 0.0
            random.shuffle(train_data)

            # for every instance, we optimize the loss of the corresponding task
            for (word_indices, char_indices), task2label_id_seq in train_data:
                # get the concatenated word and char-based features for every
                # word in the sequence
                features = self.get_word_char_features(word_indices, char_indices)
                for task, y in task2label_id_seq.items():
                    if task in [POS, CHUNK, NER, SRL]:
                        output, penalty = self.predict(features, task, train=True)
                    else:
                        raise NotImplementedError('Task %s has not been '
                                                  'implemented yet.' % task)
                    loss = dynet.esum([pick_neg_log(pred, gold) for pred, gold
                                       in zip(output, y)])
                    lv = loss.value()
                    # sum the loss and the subspace constraint penalty
                    combined_loss = loss + dynet.parameter(
                        self.constraint_weight_param, update=False) * penalty
                    total_loss += lv
                    total_penalty += penalty.value()
                    total_predicted += len(output)
                    task2total_loss[task] += lv
                    task2total_predicted[task] += len(output)
                    task2num_updates[task] += 1

                    # back-propagate through the combined loss
                    combined_loss.backward()
                    trainer.update()
                bar.next()
                num_iterations += 1

            print("\nEpoch %d. Total loss: %.3f. Total penalty: %.3f. Losses: "
                  % (epoch, total_loss / total_predicted,
                     total_penalty / total_predicted), end='', flush=True)
            for task in task2total_loss.keys():
                print('%s: %.3f. ' % (task, task2total_loss[task] /
                                      task2total_predicted[task]),
                      end='', flush=True)
            print('', flush=True)

            # evaluate after every epoch
            dev_acc = self.evaluate(dev_X, dev_Y)

            if dev_acc > best_dev_acc:
                print('Main task %s dev acc %.4f is greater than best dev acc '
                      '%.4f...' % (self.main_task, dev_acc, best_dev_acc),
                      flush=True)
                best_dev_acc = dev_acc
                num_epochs_no_improvement = 0
                print('Saving model to directory %s...' % self.model_dir,
                      flush=True)
                self.save()
            else:
                print('Main task %s dev acc %.4f is lower than best dev acc '
                      '%.4f...' % (self.main_task, dev_acc, best_dev_acc),
                      flush=True)
                num_epochs_no_improvement += 1
            if num_epochs_no_improvement == patience:
                print('Early stopping...', flush=True)
                print('Loading the best performing model from %s...'
                      % self.model_dir, flush=True)
                self.model.load(self.model_file)
                break

    def predict(self, features, task_name, train=False):
        """
        Steps through the computation graph and obtains predictions for the
        provided input features.
        :param features: a list of concatenated word and character-based
                         embeddings for every word in the sequence
        :param task_name: the name of the task that should be predicted
        :param train: if the model is training; apply noise in this case
        :return output: the output predictions
                penalty: the summed subspace penalty (0 if no constraint)
        """
        if train:  # only do at training time
            features = [dynet.noise(fe, self.noise_sigma) for fe in
                        features]

        output_expected_at_layer = self.predictors['task_expected_at'][task_name]
        output_expected_at_layer -= 1  # remove 1 as layers are 0-indexed

        # only if we use cross-stitch we have a layer for each task;
        # otherwise we just have one layer for all tasks
        num_layers = self.h_layers
        num_task_layers = len(self.predictors['inner'][0])
        inputs = [features] * num_task_layers
        inputs_rev = [features] * num_task_layers

        # similarly, with cross-stitching, we have multiple output layers
        target_task_id = self.task_names.index(
            task_name) if self.cross_stitch else 0

        # collect the forward and backward sequences for each task at every
        # layer for the layer connection units
        layer_forward_sequences = []
        layer_backward_sequences = []
        penalty = dynet.parameter(self.subspace_penalty, update=False)
        for i in range(0, num_layers):
            forward_sequences = []
            backward_sequences = []
            for j in range(num_task_layers):
                predictor = self.predictors['inner'][i][j]
                forward_sequence, backward_sequence = predictor.predict_sequence(
                    inputs[j], inputs_rev[j])
                if i > 0 and self.activation:
                    # activation between LSTM layers
                    forward_sequence = [self.activation(s) for s in
                                        forward_sequence]
                    backward_sequence = [self.activation(s) for s in
                                         backward_sequence]
                forward_sequences.append(forward_sequence)
                backward_sequences.append(backward_sequence)

                if self.num_subspaces == 2 and self.constraint_weight != 0:
                    # returns a list per layer, i.e. here a list with one item
                    lstm_parameters = \
                        predictor.builder.get_parameter_expressions()[0]

                    # lstm parameters consists of these weights:
                    # Wix,Wih,Wic,bi,Wox,Woh,Woc,bo,Wcx,Wch,bc
                    for param_idx in range(len(lstm_parameters)):
                        if param_idx in self.constrain_matrices:
                            W = lstm_parameters[param_idx]
                            W_shape = np.array(W.value()).shape

                            # split matrix into its two subspaces
                            W_subspaces = dynet.reshape(W, (
                                self.num_subspaces, W_shape[0] / float(
                                    self.num_subspaces), W_shape[1]))
                            subspace_1, subspace_2 = W_subspaces[0], W_subspaces[1]

                            # calculate the matrix product of the two matrices
                            matrix_product = dynet.transpose(
                                subspace_1) * subspace_2

                            # take the squared Frobenius norm by squaring
                            # every element and then summing them
                            squared_frobenius_norm = dynet.sum_elems(
                                dynet.square(matrix_product))
                            penalty += squared_frobenius_norm

            if self.cross_stitch:
                # takes as input a list of input lists and produces a list of
                # outputs where the index indicates the task
                forward_sequences = self.predictors['cross_stitch'][
                    i].stitch(forward_sequences)
                backward_sequences = self.predictors['cross_stitch'][
                    i].stitch(backward_sequences)

            inputs = forward_sequences
            inputs_rev = backward_sequences
            layer_forward_sequences.append(forward_sequences)
            layer_backward_sequences.append(backward_sequences)

            if i == output_expected_at_layer:
                output_predictor = \
                    self.predictors['output_layers_dict'][task_name]

                # get the forward/backward states of all task layers
                task_forward_sequences = [
                    layer_seq_list[target_task_id] for
                    layer_seq_list in layer_forward_sequences]
                task_backward_sequences = [
                    layer_seq_list[target_task_id] for
                    layer_seq_list in layer_backward_sequences]

                if self.layer_connect == STITCH:
                    # stitch the forward and backward sequences together
                    forward_inputs = \
                        self.predictors['layer_stitch'][
                            target_task_id].stitch(task_forward_sequences)
                    backward_inputs = \
                        self.predictors['layer_stitch'][
                            target_task_id].stitch(task_backward_sequences)
                elif self.layer_connect == SKIP:
                    # use skip connections
                    forward_inputs = [dynet.esum(list(layer_states))
                                      for layer_states in
                                      zip(*task_forward_sequences)]
                    backward_inputs = [dynet.esum(list(layer_states)) for
                                       layer_states in
                                       zip(*task_backward_sequences)]
                else:
                    # otherwise just use the sequences from the last layer
                    forward_inputs = forward_sequences[
                        target_task_id]
                    backward_inputs = backward_sequences[
                        target_task_id]

                if self.layer_connect == CONCAT:
                    layer_concatenated = []
                    # concatenate forward and backward states of layers
                    for fwd_seqs, bwd_seqs in zip(task_forward_sequences,
                                                  task_backward_sequences):
                        layer_concatenated.append(
                            [dynet.concatenate([f, b]) for f, b in
                             zip(fwd_seqs, reversed(bwd_seqs))])
                    # concatenate the states of all the task layers
                    concat_layer = [
                        dynet.concatenate(list(layer_states)) for
                        layer_states in zip(*layer_concatenated)]
                else:
                    concat_layer = [dynet.concatenate([f, b]) for f, b in
                                    zip(forward_inputs,
                                        reversed(backward_inputs))]

                if train and self.noise_sigma > 0.0:
                    concat_layer = [dynet.noise(fe, self.noise_sigma) for fe in
                                    concat_layer]

                output = output_predictor.predict_sequence(concat_layer)
                return output, penalty
        raise Exception('Error: This place should not be reached.')

    def evaluate(self, test_X, test_Y):
        """
        Computes accuracy on a test file.
        :param test_X: the test data; a list of (word_ids, char_ids) tuples
        :param test_Y: labels; a list of task-to-label sequence mappings
        :return accuracy on the test file
        """
        dynet.renew_cg()
        if self.cross_stitch:
            for layer_num in range(self.h_layers):
                alphas = dynet.parameter(
                    self.predictors['cross_stitch'][layer_num].alphas).value()
                print('Cross-stitch unit values at layer %d.' % layer_num,
                      end=' ', flush=True)
                if self.num_subspaces > 1:
                    print(np.array(alphas).flatten())
                else:
                    for i, task_i in enumerate(self.task_names):
                        for j, task_j in enumerate(self.task_names):
                            print('%s-%s: %3f.' % (task_i, task_j,
                                                   alphas[i][j]),
                                  end=' ', flush=True)
                print('')
        if self.layer_connect == STITCH:
            for task_id, task_name in enumerate(self.task_names):
                betas = dynet.parameter(
                    self.predictors['layer_stitch'][task_id].betas).value()
                print('Layer-stitch unit values for task %s: %s.'
                      % (task_name, ', '.join(['%.3f' % b for b in betas])),
                      flush=True)
            print('Note: Without cross-stitching, we only use the first '
                  'layer-stitch units due to hard parameter-sharing.')

        task2stats = {task: {'correct': 0, 'total': 0} for task
                      in self.task_names}
        for i, ((word_indices, word_char_indices), task2label_id_seq)\
                in enumerate(zip(test_X, test_Y)):
            for task, label_id_seq in task2label_id_seq.items():
                features = self.get_word_char_features(word_indices,
                                                       word_char_indices)
                output, _ = self.predict(features, task, train=False)
                predicted_label_indices = [np.argmax(o.value()) for o in output]
                task2stats[task]['correct'] += sum(
                    [1 for (predicted, gold) in zip(predicted_label_indices,
                                                    label_id_seq)
                     if predicted == gold])
                task2stats[task]['total'] += len(label_id_seq)

        for task, stats in task2stats.items():
            if stats['total'] == 0:
                print('No test examples available for task %s. Continuing...'
                      % task)
            else:
                print('Task: %s. Acc: %.4f. Correct: %d. Total: %d.'
                      % (task, stats['correct'] / stats['total'],
                         stats['correct'], stats['total']), flush=True)
        if task2stats[self.main_task]['total'] == 0:
            print('No test examples available for main task %s. Continuing...'
                  % self.main_task)
            return 0.
        return task2stats[self.main_task]['correct'] / task2stats[
            self.main_task]['total']

    def get_word_char_features(self, word_indices, char_indices):
        """
        Produce word and character features that can be used as input for the
        predictions.
        :param word_indices: a list of word indices
        :param char_indices: a list of lists of character ids for each token
        :return: a list of embeddings where each embedding is the
                 concatenation of word embedding and character embeddings
        """
        dynet.renew_cg()  # new graph

        char_emb = []
        rev_char_emb = []
        # get representation for words
        for chars_of_token in char_indices:
            # use last state as word representation
            last_state = self.char_rnn.predict_sequence(
                [self.cembeds[c] for c in chars_of_token])[-1]
            rev_last_state = self.char_rnn.predict_sequence(
                [self.cembeds[c] for c in reversed(chars_of_token)])[-1]
            char_emb.append(last_state)
            rev_char_emb.append(rev_last_state)

        wfeatures = [self.wembeds[w] for w in word_indices]
        features = [dynet.concatenate([w, c, rev_c]) for w, c, rev_c in
                    zip(wfeatures, char_emb, reversed(rev_char_emb))]
        return features


def pick_neg_log(pred, gold):
    """Get the negative log-likelihood of the predictions."""
    return -dynet.log(dynet.pick(pred, gold))
