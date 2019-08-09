import os
import sys
import logging
import time
from copy import deepcopy
from datetime import timedelta
import torch
from torchtext.vocab import GloVe, Vectors


class LogFormatter():

    def __init__(self):
        self.start_time = time.time()

    def format(self, record):
        elapsed_seconds = round(record.created - self.start_time)

        prefix = "%s - %s - %s" % (
            record.levelname,
            time.strftime('%x %X'),
            timedelta(seconds=elapsed_seconds)
        )
        message = record.getMessage()
        message = message.replace('\n', '\n' + ' ' * (len(prefix) + 3))
        return "%s - %s" % (prefix, message) if message else ''


def create_logger(filepath, rank):
    """
    Create a logger.
    Use a different log file for each process.
    """
    # create log formatter
    log_formatter = LogFormatter()

    # create file handler and set level to debug
    if filepath is not None:
        if rank > 0:
            filepath = '%s-%i' % (filepath, rank)
        file_handler = logging.FileHandler(filepath, "a")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(log_formatter)

    # create console handler and set level to info
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(log_formatter)

    # create logger and set level to debug
    logger = logging.getLogger()
    logger.handlers = []
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    if filepath is not None:
        logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    # reset logger elapsed time
    def reset_time():
        log_formatter.start_time = time.time()
    logger.reset_time = reset_time

    return logger


def init_logger(params):
    command = ["python", sys.argv[0]]
    for x in sys.argv[1:]:
        if x.startswith('--'):
            assert '"' not in x and "'" not in x
            command.append(x)
        else:
            assert "'" not in x
            command.append("'%s'" % x)
    command = ' '.join(command)
    logger = create_logger(os.path.join(params.dump_path, 'train.log'), rank=getattr(params, 'global_rank', 0))
    logger.info("============ Initialized logger ============")
    logger.info("\n".join("%s: %s" % (k, str(v))
                          for k, v in sorted(dict(vars(params)).items())))
    logger.info("Running command: %s" % command)
    logger.info("")
    logger_scores = create_logger(os.path.join(params.dump_path, 'scores.log'), rank=getattr(params, 'global_rank', 0))

    return logger, logger_scores


def to_cuda(*args):
    """
    Move tensors to CUDA.
    """
    return [None if x is None else x.cuda() for x in args]


def load_w2v(data_path, dim_embedding):
    # glove = GloVe(dim=dim_embedding)
    glove = Vectors(data_path, ".")
    vocab_size = len(glove.stoi)
    word2idx = glove.stoi
    weight = deepcopy(glove.vectors)
    weight = torch.cat((weight, weight.new(1, dim_embedding).fill_(0.0)), dim=0)
    word2idx['UNK'] = vocab_size
    return word2idx, weight


if __name__ == "__main__":
    word2idx, weight = load_w2v('data/glove.6B.300d.txt', 300)
    print(weight.size())
    print(weight[len(word2idx)-1].tolist())
