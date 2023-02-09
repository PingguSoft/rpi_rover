import sys
import logging

def init_log(name=None, log_level=logging.INFO):
    logger = logging.getLogger(name)

    # only root has a log handler
    if name is None:
        hnd = logging.StreamHandler(sys.stdout)
        hnd.setFormatter(logging.Formatter('%(asctime)s.%(msecs)03d [%(process)5d-%(thread)-5d] [%(levelname)8s] ' +
                                           '[%(filename)20s:%(lineno)6d] %(funcName)25s - %(message)s',
                                            '%H:%M:%S'))
        hnd.setLevel(log_level)
        logger.addHandler(hnd)
    logger.setLevel(log_level)

    return logger
