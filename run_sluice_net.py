"""
Main script to run Sluice Networks.
"""
import argparse
import os
import random
import sys

import numpy as np

import dynet

from constants import TASK_NAMES, DOMAINS, MODEL_FILE, PARAMS_FILE, \
    BALANCED, IMBALANCED, STITCH, CONCAT, SKIP, NONE, SGD, ADAM
from sluice_net import SluiceNetwork, load
import utils


def check_activation_function(arg):
    """Checks allowed argument for --ac option."""
    try:
        functions = [dynet.rectify, dynet.tanh]
        functions = {function.__name__: function for function in functions}
        functions['None'] = None
        return functions[str(arg)]
    except:
        raise argparse.ArgumentTypeError(
            'String {} does not match required format'.format(arg, ))


def main(args):
    if args.dynet_seed:
        print('>>> using seed: ', args.dynet_seed, file=sys.stderr, flush=True)
        np.random.seed(args.dynet_seed)
        random.seed(args.dynet_seed)

    if args.c_in_dim == 0:
        print('no character embeddings', file=sys.stderr, flush=True)

    # check if folder exists
    if not os.path.exists(args.model_dir):
        print('Creating model directory %s...' % args.model_dir, flush=True)
        os.makedirs(args.model_dir)
    print('Note: Use different model-dir paths for different runs. Otherwise '
          'files might be overwritten.', file=sys.stderr, flush=True)
    if not os.path.exists(args.log_dir):
        print('Creating log directory %s...' % args.log_dir, flush=True)
        os.makedirs(args.log_dir)

    for dir_path in [args.train_dir, args.dev_dir, args.test_dir]:
        assert os.path.exists(dir_path), 'Error: %s does not exist.' % dir_path

    if len(args.test) < 2:
        print('No or only one test domain is being used. Model can be '
              'evaluated on all available domains.', file=sys.stderr, flush=True)

    assert args.num_subspaces > 1 or args.constraint_weight == 0,\
        'Error: More than 1 subspace necessary for subspace constraint.'
    assert args.constrain_matrices or args.constraint_weight == 0,\
        ('Error: When subspace constraint is specified, indices of matrices '
         'to be constrained need to be provided.')
    assert args.cross_stitch or args.constraint_weight == 0,\
        'Error: Subspace constraint only works with cross-stitch units.'
    assert len(args.task_names) > 1 or not args.cross_stitch,\
        'Error: Cross-stitch units only work in an MTL setting.'
    assert args.h_layers > 1 or args.layer_connect != STITCH,\
        'Error: Layer-stitch units only work with more than one layer.'
    assert all([pred == args.h_layers for pred in args.pred_layer]) or not \
        args.layer_connect == STITCH, ('Error: All predictions should take place at '
                            'final layer if layer-stitch units are used.')
    assert not any(x in args.constrain_matrices for x in [3, 7, 10]),\
        ('Error: Index 3/7/10 belongs to bias vectors, which are not '
         'constrained.')

    assert len(args.task_names) == len(args.pred_layer),\
        ('Error: %d task names provided but %d ids for pred layers given.' %
         (len(args.task_names), len(args.pred_layer)))

    print('Using %s layer connections before FC layer...' % args.layer_connect,
          flush=True)
    if args.constraint_weight != 0:
        print('Using subspace constraint with constraint weight %.4f...'
              % args.constraint_weight)
        print('Using squared Frobenius norm constraint on LSTM matrices with '
              'ids %s...' %
              ', '.join(['%d' % d for d in args.constrain_matrices]))
    print('Tasks used: %s' % ', '.join(args.task_names), flush=True)

    if args.load:
        assert os.path.exists(args.model_dir),\
            ('Error: Trying to load the model but %s does not exist.' %
             args.model_dir)
        print('Loading model from directory %s...' % args.model_dir)
        params_file = os.path.join(args.model_dir, PARAMS_FILE)
        model_file = os.path.join(args.model_dir, MODEL_FILE)
        model = load(params_file, model_file, args)
    else:
        model = SluiceNetwork(args.in_dim,
                              args.h_dim,
                              args.c_in_dim,
                              args.h_layers,
                              args.pred_layer,
                              args.model_dir,
                              embeds_file=args.embeds,
                              activation=args.ac,
                              lower=args.lower,
                              noise_sigma=args.sigma,
                              task_names=args.task_names,
                              cross_stitch=args.cross_stitch,
                              layer_connect=args.layer_connect,
                              num_subspaces=args.num_subspaces,
                              constraint_weight=args.constraint_weight,
                              constrain_matrices=args.constrain_matrices,
                              cross_stitch_init_scheme=
                              args.cross_stitch_init_scheme,
                              layer_stitch_init_scheme=
                              args.layer_stitch_init_scheme)
        model.fit(args.train, args.epochs, args.patience, args.opt,
                  train_dir=args.train_dir, dev_dir=args.dev_dir)

    for i, test_domain in enumerate(args.test):
        print('\nTesting on domain %s...' % test_domain)
        test_X, test_Y, _, _, _, _, _ = utils.get_data(
            [test_domain], model.task_names, model.word2id, model.char2id,
            model.task2tag2idx, data_dir=args.test_dir, train=False)

        test_accuracy = model.evaluate(test_X, test_Y)
        print('Train: %s. Test: %s.' % (args.train, test_domain), flush=True)
        print('Main task: %s. Test accuracy: %.4f'
              % (model.main_task, test_accuracy), flush=True)

        log_file = os.path.join(args.log_dir, 'log.txt')
        utils.log_score(log_file, args.train, test_domain, test_accuracy,
                        args.task_names, args.h_layers, args.cross_stitch,
                        args.layer_connect, args.num_subspaces,
                        args.constraint_weight, args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Run the Sluice Network',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # DyNet parameters
    parser.add_argument('--dynet-autobatch', type=int,
                        help='use auto-batching (1) (should be first argument)')
    parser.add_argument('--dynet-seed', type=int, help='random seed for DyNet')
    parser.add_argument('--dynet-mem', type=int, help='memory for DyNet')

    # domains, tasks, and paths
    parser.add_argument('--train', required=True, choices=DOMAINS,
                        help='the domain for training the model')
    parser.add_argument('--test', nargs='*', choices=DOMAINS,
                        help='the domains used for testing the model')
    parser.add_argument('--train-dir', required=True,
                        help='the directory containing the training data')
    parser.add_argument('--dev-dir', required=True,
                        help='the directory containing the development data')
    parser.add_argument('--test-dir', required=True,
                        help='the directory containing the test data')
    parser.add_argument('--load', action='store_true',
                        help='load the pre-trained model')
    parser.add_argument('--task-names', nargs='+', required=True,
                        choices=TASK_NAMES,
                        help='the names of the tasks (main task is first)')
    parser.add_argument('--model-dir', required=True,
                        help='directory where to save model and param files')
    parser.add_argument('--log-dir', required=True,
                        help='the directory where the results should be logged')

    # model-specific hyperparameters
    parser.add_argument('--pred-layer', nargs='+', type=int, default=[1],
                        help='layer of predictions for each task')
    parser.add_argument('--in-dim', type=int, default=64,
                        help='input dimension [default: 64]')
    parser.add_argument('--c-in-dim', type=int, default=100,
                        help='input dim for char embeddings [default:100]')
    parser.add_argument('--h-dim', type=int, default=100,
                        help='hidden dimension [default: 100]')
    parser.add_argument('--h-layers', type=int, default=1,
                        help='number of stacked LSTMs [default: 1=no stacking]')
    parser.add_argument('--lower', action='store_true',
                        help='lowercase words (not used)')
    parser.add_argument('--embeds', help='word embeddings file')
    parser.add_argument('--sigma', help='noise sigma', default=0.2, type=float)
    parser.add_argument('--ac', default='tanh',
                        help='activation function [rectify, tanh, ...]',
                        type=check_activation_function)
    parser.add_argument('--opt', '--optimizer', default=SGD,
                        choices=[SGD, ADAM],
                        help='trainer [sgd, adam] default: sgd')

    # training hyperparameters
    parser.add_argument('--epochs', type=int, default=30,
                        help='training epochs [default: 30]')
    parser.add_argument('--patience', default=1, type=int,
                        help='patience for early stopping')

    parser.add_argument('--cross-stitch', action='store_true',
                        help='use cross-stitch units between LSTM layers')
    parser.add_argument('--layer-connect', default=NONE,
                        choices=[STITCH, CONCAT, SKIP, NONE],
                        help='type of layer connection before FC layer that '
                             'should be used')
    parser.add_argument('--num-subspaces', default=1, type=int, choices=[1, 2],
                        help='the number of subspaces for cross-stitching; '
                             'only 1 (no subspace) or 2 allowed currently')
    parser.add_argument('--constraint-weight', type=float, default=0.,
                        help='weighting factor for orthogonality constraint on '
                             'cross-stitch subspaces; 0 = no constraint')
    parser.add_argument('--constrain-matrices', type=int, nargs='+',
                        default=[1, 2],
                        help='the indices of the LSTM matrices that should be '
                             'constrained; indices correspond to: Wix,Wih,Wic,'
                             'bi,Wox,Woh,Woc,bo,Wcx,Wch,bc. Best indices so '
                             'far: [1, 2] http://dynet.readthedocs.io/en/latest/python_ref.html#dynet.LSTMBuilder.get_parameter_expressions)')
    parser.add_argument('--cross-stitch-init-scheme', type=str,
                        default=BALANCED, choices=[IMBALANCED, BALANCED],
                        help='which initialisation scheme to use for the '
                             'alpha matrix - currently available: imbalanced '
                             'and balanced (which sets all to '
                             '1/(num_tasks*num_subspaces)). Only available '
                             'with subspaces.')
    parser.add_argument('--layer-stitch-init-scheme', type=str,
                        default=IMBALANCED,
                        choices=[BALANCED, IMBALANCED],
                        help='initialisation scheme for layer-stitch units; '
                             'default: imbalanced (.9) for last layer weights;'
                             'other choice: balanced (1. / num_layers).')
    args = parser.parse_args()
    main(args)
