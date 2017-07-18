"""
Utility methods for data processing.
"""
import os
from glob import glob
import itertools

from constants import NUM, NUMBERREGEX, POS, NER, SRL, CHUNK, UNK, WORD_START,\
    WORD_END


class ConllEntry:
    """Class representing an entry, i.e. word and its annotations in CoNLL
    """
    def __init__(self, id, form, tasks, pos=None, ner_tag=None, srl_tag=None,
                 chunk=None):
        """
        Initializes a CoNLL entry.
        :param id: the id of a word
        :param form: the word form
        :param tasks: the tasks for which this entry has annotations
        :param pos: the part-of-speech tag
        :param ner_tag: the NER tag
        :param srl_tag: the SRL tag
        :param chunk: the chunk tag
        """
        self.id = id  # word index; integer starting with 1
        self.form = form  # the word form

        # normalize form (lower-cased and numbers replaced with NUM)
        self.norm = normalize(form)
        self.pos = pos  # language-specific POS
        self.ner_tag = ner_tag
        self.srl_tag = srl_tag
        self.tasks = tasks
        self.chunk = chunk


def normalize(word):
    """Normalize a word by lower-casing it or replacing it if it is a number."""
    return NUM if NUMBERREGEX.match(word) else word.lower()


def load_embeddings_file(file_name, sep=" ", lower=False):
    """Loads a word embedding file."""
    word2vec = {}
    for line in open(file_name):
        fields = line.split(sep)
        vec = [float(x) for x in fields[1:]]
        word = fields[0]
        if lower:
            word = word.lower()
        word2vec[word] = vec
    print('Loaded pre-trained embeddings of size: {} (lower: {})'
          .format(len(word2vec.keys()), lower))
    return word2vec, len(word2vec[word])


def read_conll_file(file_path, tasks=None, verbose=False):
    """
    Reads in an OntoNotes 5.0 file in CoNLL format, i.e.
    bc/cctv/00/cctv_0001   0    0              We   PRP   (TOP(S(NP*)       -    -   -   Speaker#1       *        (ARG0*)        *   -
    bc/cctv/00/cctv_0001   0    1    respectfully    RB       (ADVP*)       -    -   -   Speaker#1       *    (ARGM-MNR*)        *   -
    bc/cctv/00/cctv_0001   0    2          invite    VB         (VP*    invite  01   3   Speaker#1       *           (V*)        *   -
    bc/cctv/00/cctv_0001   0    3             you   PRP         (NP*)       -    -   -   Speaker#1       *        (ARG1*)   (ARG0*)  -
    Sentences are separated by newlines.
    :param file_path: path to the file to read
    :param tasks: the tasks associated with the file
    :param verbose: whether to print more information about reading
    :return: generator of instances ((list of  words, list of tags) pairs)
    """
    if tasks is None:
        tasks = [POS, NER, SRL]

    if verbose:
        print('Reading CoNLL file %s...' % file_path)
    with open(file_path, encoding='utf-8') as f:
        conll_entries = []
        # so we keep track of previous tag for conversion to BIO notation
        prev_ner_tag_type, ner_within_tag = 'O', False
        for line in f:
            # check if line is newline
            if line == '\n' or line.startswith('#'):  # beginning of document
                if len(conll_entries) > 1:
                    yield conll_entries
                conll_entries = []
            else:
                # legend here: http://cemantix.org/data/ontonotes.html
                doc_id, part_num, word_id, word_form, postag, parse_bit, lemma,\
                    frameset_id, word_sense, speaker, ner_tag, *rest = \
                    line.strip().split()
                srl_tags, srl_bio_tag = [], None

                # we only use the verb identification tags (not the ARG tags)
                # for SRL
                if len(rest) == 1:
                    coref = rest[0]
                elif len(rest) == 1:
                    temp_srl_tag, coref = rest
                    srl_tags.append(temp_srl_tag)
                else:
                    srl_tags, coref = rest[0:-1], rest[-1]
                if len(srl_tags) > 0:
                    srl_bio_tag = 'I-V' if '(V*)' in srl_tags else 'O'
                else:
                    srl_bio_tag = None
                ner_bio_tag, prev_ner_tag_type, ner_within_tag = tag2BIO_tag(
                    ner_tag, prev_ner_tag_type, ner_within_tag)
                conll_entries.append(
                    ConllEntry(int(word_id), word_form, tasks, pos=postag,
                               ner_tag=ner_bio_tag, srl_tag=srl_bio_tag))
        if len(conll_entries) > 1:
            yield conll_entries


def tag2BIO_tag(tag, prev_tag_type, within_tag):
    """Convert a tag to BIO notation."""
    if within_tag:
        tag_type = prev_tag_type
        bio_tag = 'I-' + tag_type
        within_tag = False if tag.endswith(')') else True
    elif tag.startswith('('):
        tag_type = tag.strip('(*)')
        bio_tag = 'B-' + tag_type
        within_tag = False if tag.endswith(')') else True
    else:
        tag_type = 'O'
        bio_tag = tag_type
    return bio_tag, tag_type, within_tag


def read_file(file_path, tasks, verbose=False):
    """
    Reads a file for chunking and returns it as a generator of CoNLL entries.
    :param file_path:
    :return:
    """
    assert len(tasks) == 1 and tasks[0] == CHUNK, 'Error: read_file only used for chunking so far.'
    if verbose:
        print('Reading file %s...' % file_path)
    with open(file_path, encoding='utf-8') as f:
        conll_entries = []
        for i, line in enumerate(f):
            if line == '\n':
                if len(conll_entries) > 0:
                    yield conll_entries
                conll_entries = []
            else:
                try:
                    word, label = line.strip().split('\t')
                except ValueError:
                    print('Error at line %d, %s:' % (i, file_path), line)
                    conll_entries = []
                    continue
                if label is None:
                    print()
                conll_entries.append(ConllEntry(len(conll_entries)+1, word, tasks, chunk=label))
        if len(conll_entries) > 0:
            yield conll_entries


def get_data(domains, task_names, word2id=None, char2id=None,
             task2label2id=None, data_dir=None, train=True, verbose=False):
    """
    :param domains: a list of domains from which to obtain the data
    :param task_names: a list of task names
    :param word2id: a mapping of words to their ids
    :param char2id: a mapping of characters to their ids
    :param task2label2id: a mapping of tasks to a label-to-id dictionary
    :param data_dir: the directory containing the data
    :param train: whether data is used for training (default: True)
    :param verbose: whether to print more information re file reading
    :return X: a list of tuples containing a list of word indices and a list of
               a list of character indices;
            Y: a list of dictionaries mapping a task to a list of label indices;
            org_X: the original words; a list of lists of normalized word forms;
            org_Y: a list of dictionaries mapping a task to a list of labels;
            word2id: a word-to-id mapping;
            char2id: a character-to-id mapping;
            task2label2id: a dictionary mapping a task to a label-to-id mapping.
    """
    X = []
    Y = []
    org_X = []
    org_Y = []

    # for training, we initialize all mappings; for testing, we require mappings
    if train:
        assert word2id is None, ('Error: Word-to-id mapping should not be '
                                 'provided for training.')
        assert char2id is None, ('Error: Character-to-id mapping should not '
                                 'be provided for training.')

        # create word-to-id, character-to-id, and task-to-label-to-id mappings
        word2id, char2id = {}, {}
        task2label2id = {task: {} for task in task_names}

        # set the indices of the special characters
        word2id[UNK] = 0  # unk word / OOV
        char2id[UNK] = 0  # unk char
        char2id[WORD_START] = 1  # word start
        char2id[WORD_END] = 2  # word end index

        # manually add tags only available in some domains for POS tagging
        if POS in task_names:
            for label in ['NFP', 'ADD', '$', '', 'CODE', 'X', 'VERB']:
                task2label2id[POS][label] = len(task2label2id[POS])
    else:
        assert word2id is not None, 'Error: Word-to-id mapping is required.'
        assert char2id is not None, 'Error: Char-to-id mapping is required.'
        assert task2label2id is not None, 'Error: Task mapping is required.'
        assert UNK in word2id
        assert UNK in char2id
        assert WORD_START in char2id
        assert WORD_END in char2id

    for domain in domains:
        num_sentences = 0
        num_tokens = 0

        file_reader = iter(())
        domain_path = os.path.join(data_dir, 'data', 'english',
                                   'annotations', domain)
        assert os.path.exists(domain_path), ('Domain path %s does not exist.'
                                             % domain_path)
        # read files in the domain path and add the file reader to the generator
        if POS in task_names or SRL in task_names or NER in task_names:
            # POS tagging, SRL, and NER use the same files
            for file_path in itertools.chain.from_iterable(
                    glob(os.path.join(x[0], '*.gold_conll'))
                    for x in os.walk(domain_path)):
                file_reader = itertools.chain(
                    file_reader, read_conll_file(file_path, verbose=verbose))
        if CHUNK in task_names:
            # we have separate files with chunking annotations
            for file_path in itertools.chain.from_iterable(
                    (glob(os.path.join(x[0], '*.chunks'))
                     for x in os.walk(domain_path))):
                file_reader = itertools.chain(
                    file_reader, read_file(file_path, [CHUNK], verbose=verbose))

        # the file reader should returns a list of CoNLL entries; we then get
        # the relevant labels for each task
        for sentence_idx, conll_entries in enumerate(file_reader):
            num_sentences += 1
            sentence_word_indices = []  # sequence of word indices
            sentence_char_indices = []  # sequence of char indices
            # keep track of the label indices and labels for each task
            sentence_task2label_indices = {}
            sentence_task2labels = {}

            # keep track of the original word forms
            org_X.append([conll_entry.norm for conll_entry in conll_entries])

            for i, conll_entry in enumerate(conll_entries):
                num_tokens += 1
                word = conll_entry.norm

                # add words and chars to the mapping
                if train and word not in word2id:
                    word2id[word] = len(word2id)
                sentence_word_indices.append(word2id.get(word, word2id[UNK]))

                chars_of_word = [char2id[WORD_START]]
                for char in word:
                    if train and char not in char2id:
                        char2id[char] = len(char2id)
                    chars_of_word.append(char2id.get(char, char2id[UNK]))
                chars_of_word.append(char2id[WORD_END])
                sentence_char_indices.append(chars_of_word)

                # get the labels for the task if we have annotations
                for task in task2label2id.keys():
                    if task in conll_entry.tasks:
                        if task == POS:
                            label = conll_entry.pos
                        elif task == CHUNK:
                            label = conll_entry.chunk
                        elif task == NER:
                            label = conll_entry.ner_tag
                        elif task == SRL:
                            # not all sentences have SRL annotation
                            if conll_entry.srl_tag is None:
                                continue
                            label = conll_entry.srl_tag
                        else:
                            raise NotImplementedError('Label for task %s is not'
                                                      ' implemented.' % task)
                        if task not in sentence_task2label_indices:
                            sentence_task2label_indices[task] = []
                        if task not in sentence_task2labels:
                            sentence_task2labels[task] = []
                        assert label is not None, ('Label is None for task '
                                                   '%s.' % task)
                        if not train and label not in task2label2id[task]:
                            print('Error: Unknown label %s for task %s not '
                                  'valid during testing.' % (label, task))
                            print('Assigning id of another label as we only '
                                  'care about main task scores...')
                            task2label2id[task][label] =\
                                len(task2label2id[task]) - 1
                        if train and label not in task2label2id[task]:
                            task2label2id[task][label] = \
                                len(task2label2id[task])
                        sentence_task2label_indices[task].\
                            append(task2label2id[task].get(label))
                        sentence_task2labels[task].append(label)

            if len(task_names) == 1 and task_names[0] == SRL:
                if len(sentence_task2label_indices) == 0:
                    continue
            assert len(sentence_task2label_indices) > 0,\
                'Error: No label/task available for entry.'
            X.append((sentence_word_indices, sentence_char_indices))
            Y.append(sentence_task2label_indices)
            org_Y.append(sentence_task2labels)

        assert num_sentences != 0 and num_tokens != 0, ('No data read for '
                                                        '%s.' % domain)
        print('Number of sentences: %d. Number of tokens: %d.'
              % (num_sentences, num_tokens))
        print("%s sentences %s tokens" % (num_sentences, num_tokens))
        print("%s w features, %s c features " % (len(word2id), len(char2id)))

    for task, label2id in task2label2id.items():
        print('Task %s. Labels: %s' % (task, [l for l in label2id.keys()]))

    assert len(X) == len(Y)
    return X, Y, org_X, org_Y, word2id, char2id, task2label2id


def log_score(log_file, src_domain, trg_domain, accuracy, task_names,
              h_layers, cross_stitch, layer_connect, num_subspaces,
              constraint_weight, args):
    with open(log_file, 'a') as f:
        f.write('%s->%s\t%.4f\t%s\t%d\tcross_stitch=%s\tlayer=%s\t%d'
                '\tconstraint_weight=%.4f\t%s\n'
                % (src_domain, trg_domain, accuracy, ', '.join(task_names),
                   h_layers, cross_stitch, layer_connect,
                   num_subspaces, constraint_weight,
                   ' '.join(['%s=%s' % (arg, str(getattr(args, arg)))
                             for arg in vars(args)])))
