import logging


class LogModule:
    logger = logging.getLogger(__name__)
    c_handler = logging.StreamHandler()
    f_handler = logging.FileHandler('file.log')
    logging.root.setLevel(logging.DEBUG)

    c_format = logging.Formatter('%(name)s - %(message)s to console')
    f_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s to file')
    c_handler.setFormatter(c_format)
    f_handler.setFormatter(f_format)

    logger.addHandler(c_handler)
    logger.addHandler(f_handler)

