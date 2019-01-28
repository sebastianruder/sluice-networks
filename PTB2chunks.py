"""
Script to extract chunk annotations from the original OntoNotes 5.0 data and
save the files into the file structure of the CoNLL formatted OntoNotes 5.0
files.
"""
import os
import argparse
import itertools
from glob import glob


def convertOntonotesChunks(original_folder, conll_folder):
    """
    Converts OntoNotes PTB annotations to chunk annotations. Chunk annotations
    are stored as *.chunks files in the file structure of the CoNLL formatted
    OntoNotes files. The format is the following:
    Abramov	O
    had	B-VP
    a	B-NP
    car	I-NP
    accident	I-NP
    :param original_folder: the path to the original OntoNotes 5.0 data
    :param conll_folder: the path to the CoNLL formatted OntoNotes 5.0 data
    """
    original_folder_data = os.path.join(original_folder, 'annotations')
    conll_folders_data = [os.path.join(conll_folder, 'train', 'data',
                                          'english', 'annotations'),
                             os.path.join(conll_folder, 'test', 'data',
                                          'english', 'annotations'),
                             os.path.join(conll_folder, 'development',
                                          'data', 'english', 'annotations'),
                             os.path.join(conll_folder, 'conll-2012-test',
                                          'data', 'english', 'annotations')]
    for fold in conll_folders_data + original_folder_data:
        assert os.path.exists(fold), ('Error: %s does not exist. Did you '
                                      'specify the correct original and CoNLL '
                                      'formatted OntoNotes paths?')

    # iterate through the various levels subfolders, the first one is for the
    # different domains
    for fold in os.listdir(original_folder_data):
        fold_domain = os.path.join(original_folder_data, fold)
        filelist = [glob(os.path.join(x[0], '*.parse')) for x in
                    os.walk(fold_domain)]
        print(fold_domain)

        for file in itertools.chain.from_iterable(filelist):
                string = open(file).read()  # read parse files
                filen = file.split('/')[-1]

                on_conll_folder_file = ''
                for thisf in conll_folders_data:
                    try:
                        filelist_conll =\
                            [glob(os.path.join(x[0], '*.gold_conll'))
                             for x in os.walk(os.path.join(thisf, fold))]
                        filelist_conll_flat =\
                            [f for f in itertools.chain.from_iterable(
                                filelist_conll)]
                        filelist_conll_flat_fonly =\
                            [f.split('/')[-1] for f in
                             itertools.chain.from_iterable(filelist_conll)]
                        if str(filen).replace('.parse', '.gold_conll') in \
                                filelist_conll_flat_fonly:
                            ind = filelist_conll_flat_fonly.index(
                                str(filen).replace('.parse', '.gold_conll'))

                            on_conll_folder_file = filelist_conll_flat[
                                ind].replace('.gold_conll', '.chunks')
                            break

                    except FileNotFoundError:
                        continue

                # corresponding file in the right out_folder to print to
                f_out = open(on_conll_folder_file, 'w')

                inNP, inVP, inADJP, inSBAR = False, False, False, False
                D, E = 0, 0
                for word in string.split():
                    if word[-1] == ')':
                        terminal = word.rstrip(')')
                        if inNP and E > 0:
                            if First:
                                # print(terminal+'\tB-NP', flush=True)
                                f_out.write(terminal+'\tB-NP\n')
                            else:
                                # print(terminal+'\tI-NP', flush=True)
                                f_out.write(terminal+'\tI-NP\n')
                            First = False
                        elif inVP and E > 0:
                            if First:
                                # print(terminal+'\tB-VP', flush=True)
                                f_out.write(terminal+'\tB-VP\n')
                            else:
                                # print(terminal+'\tI-VP', flush=True)
                                f_out.write(terminal+'\tI-VP\n')
                            First = False
                        elif inADJP and E > 0:
                            if First:
                                # print(terminal+'\tB-ADJP', flush=True)
                                f_out.write(terminal+'\tB-ADJP\n')
                            else:
                                # print(terminal+'\tI-ADJP', flush=True)
                                f_out.write(terminal+'\tI-ADJP\n')
                            First = False
                        elif inSBAR and E > 0:
                            if First:
                                # print(terminal+'\tB-SBAR', flush=True)
                                f_out.write(terminal+'\tB-SBAR\n')
                            else:
                                # print(terminal+'\tI-SPAR', flush=True)
                                f_out.write(terminal+'\tI-SPAR\n')
                            First = False
                        else:
                            # print(terminal+'\tO', flush=True)
                            f_out.write(terminal+'\tO\n')
                    for i in range(len(word)):
                        if word[i] == '(':
                            D += 1
                            E += 1
                        elif word[i] == ')':
                            D -= 1
                            E -= 1
                    word = word.lstrip('(').rstrip(')')
                    if E == 0:
                        inNP, inVP, inADJP, inSBAR = False, False, False, False
                    if word == 'NP':
                        inNP = True
                        inVP, inADJP, inSBAR = False, False, False
                        E = 1
                        First = True
                    elif word == 'VP':
                        inVP = True
                        inNP, inADJP, inSBAR = False, False, False
                        E = 1
                        First = True
                    elif word == 'ADJP':
                        inADJP = True
                        inNP, inVP, inSBAR = False, False, False
                        E=1
                        First=True
                    elif word == 'SBAR':
                        inSBAR = True
                        inNP, inVP, inADJP = False, False, False
                        E = 1
                        First = True
                    if D == 0:
                        #print('', flush=True)
                        f_out.write('\n')
                f_out.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Convert PTB annotations to chunk annotations.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--original-folder',
                        help='the path to the original OntoNotes 5.0 folder')
    parser.add_argument('--conll-folder',
                        help='the path to the CoNLL formatted OntoNotes folder')
    args = parser.parse_args()
    assert os.path.exists(args.original_folder), ('Error: %s does not exist.'
                                                  % args.original_folder)
    assert os.path.exists(args.conll_folder), ('Error: %s does not exist.'
                                                  % args.conll_formatted)
    convertOntonotesChunks(original_folder='original-ontonotes-5.0',
                           conll_folder='conll-formatted-ontonotes-5.0')
