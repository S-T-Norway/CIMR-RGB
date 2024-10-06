import logging 





def add(a, b, logger = None): 

    if logger is None: 
        logger = logging.getLogger(__name__)
        # Adding Null Handler to the logger as per python's official documentation 
        logger.addHandler(logging.NullHandler())

    # This will work because the root logger is configured 
    logger.info("Debug message")

    return a + b





def multiply(a, b, logger = None): 

    if logger is None: 
        logger = logging.getLogger(__name__)
        # Adding Null Handler to the logger as per python's official documentation 
        logger.addHandler(logging.NullHandler())

    logger.debug("My logger multiplication") 

    return a * b 
