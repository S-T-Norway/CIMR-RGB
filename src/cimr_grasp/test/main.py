import pathlib as pb 
import time 
import json 
import logging.config   
import functools 
import psutil 

from rgb_logging import RGBLogging  
from module1 import add, multiply





logger_config = {
    "version": 1, 
    "disable_existing_loggers": False, 
    "loggers": {
        "root": {
            "level": "DEBUG", 
            "handlers": ["stdout"]
        }
    }, 
    "handlers": {
        "stdout": {
            "class": "logging.StreamHandler", 
            "formatter": "simple", 
            "stream": "ext://sys.stdout", 
            "level": "INFO"
        }
    },  
    "formatters": { 
        "simple": { 
           "format": "%(levelname)s: %(name)s : %(message)s" 
        }, 
    } 
}










if __name__ == '__main__': 

    a = 2
    b = 1

    # Configuring `root` logger through basic config and creating custom logger with the same configuration 
    #logging.basicConfig(level = logging.DEBUG, format="%(levelname)s - %(name)s - %(message)s")
    #logger = logging.getLogger(__name__)

    # Passing in the file 
    logger_config = pb.Path("./logger_config.json")  
    rgb_logging = RGBLogging(log_config = logger_config) 
    rgb_logger  = rgb_logging.get_logger("rgb")#__name__) 


    result = add(a, b, logger = rgb_logger)
    rgb_logger.info("My info")
    rgb_logger.info(result)

    func = RGBLogging.rgb_decorated(
            decorate = True, decorator = RGBLogging.track_perf, logger = rgb_logger
            )(multiply)
    result = func(a, b)
    rgb_logger.info(result)

    print("--------")
    func = RGBLogging.rgb_decorated(decorate = False, decorator = RGBLogging.track_perf, logger = rgb_logger)(multiply)
    result = func(a, b)
    rgb_logger.info(result)







exit() 



































#from logging_config import configure_logging
#from performance_tracker import PerformanceTracker
#import module1
#import module2

# TODO: Log the memory, cpu usage and the time spent for every relevant function call. 

## Creating a logger 
#logger = logging.getLogger(__name__)
#logger.setLevel(logging.INFO)
#
## Creating a formatter 
#formatter = logging.Formatter("%(levelname)s:%(name)s: %(message)s") 
#
## Creating a file handler (that will save stuff to log file)
#file_handler = logging.FileHandler("new_test.log") 
#file_handler.setFormatter(formatter)
#
#stream_handler = logging.StreamHandler() 
#stream_handler.setFormatter(formatter)
#
#logger.addHandler(file_handler)
#logger.addHandler(stream_handler) 

#logger_config = {
#    "version": 1, 
#    "disable_existing_loggers": False, 
#    "formatters": { 
#        "simple": { 
#            "format": "%(levelname)s: %(message)s" 
#        }
#    }, 
#    "handlers": {
#        "stdout": {
#            "class": "logging.StreamHandler", 
#            "formatter": "simple", 
#            "stream": "ext://sys.stdout", 
#            "level": "DEBUG"
#        }
#    }, 
#    "loggers": {
#        "rgb": {
#            "level": "DEBUG", 
#            "handlers": ["stdout"]
#        }
#    }
#}

class RGBLogger: 
    def __init__(self, config_file = None): #pb.Path("./logger_config.json").resolve()): 

        if config_file is not None: 
            with open(config_file, "r") as file: 
                config = json.load(file) 

            self.config = logging.config.dictConfig(config)
        else: 
            self.config = None 


    def get_logger(self, name): 

        return logging.getLogger(name)


    @staticmethod 
    def rgb_decorate(use_it, decorator):
        ...

    @staticmethod 
    def profile_it(func): 

        @functools.wraps(func)
        def wrapper(*args, **kwargs): 
            ... 

        return wrapper 



import logging 

logger = logging.getLogger(__name__)


class RGBDecorators: 

    @staticmethod 
    def conditional_decorator(condition, decorator):
        """
        A factory that returns a conditional decorator.
        If `condition` is True, applies `decorator`. Otherwise, returns the original function.
        
        Parameters:
        - condition (bool): The condition to check.
        - decorator (function): The decorator to apply if the condition is True.
        """
        def wrapper(func):
            # If the condition is true, apply the decorator
            if condition:
                return decorator(func)
            # Otherwise, return the original function
            return func
        return wrapper


    @staticmethod  
    def timeit(func):
    #def timeit(self, func):

        @functools.wraps(func) # <= without this line it will write finished `wrapper` as well as the decorator we want to apply 
        def wrapper(*args, **kwargs):
            
            start_time = time.perf_counter()

            result     = func(*args, **kwargs)
            
            end_time   = time.perf_counter()
            time_taken = end_time - start_time
            print(f"Function '{func.__name__}' executed in {time_taken:.4f} seconds.")

            return result

        return wrapper
        

#def setup_logging(config_file = pb.Path("./logger_config.json").resolve()): 
#
#    with open(config_file, "r") as file: 
#        config = json.load(file) 
#
#    logging.config.dictConfig(config)


#        # This one liner configures logging 
#        #if config is not None: 
#        #    self.config = logging.config.dictConfig(config = config) 
#        #else: 
#        #    print("RGB logger was instantiated without config") 
#        self.config = logging.config.dictConfig(config = config) 
#
#    def get_me(): 
#        ... 

#setup_logging() 
#logger = logging.getLogger(__name__)

rgblogger = RGBLogger(pb.Path("./logger_config.json").resolve())  
logger = rgblogger.get_logger("cimr_rgb")


#logger.info("An info")
#logger.warning("A warning")

#@RGBLogger.timeit
@RGBDecorators.conditional_decorator(condition = True, decorator = RGBDecorators.timeit)
def divide(x, y):
    try: 
        result = x / y 
    except ZeroDivisionError: 
        logger.exception("Tried to divide by zero")
    else: 
        return result 

# Instantiating custom RGB logger 
#rgb_logging = RGBLogger(logger_config).config 
#logger = rgb_logging.getLogger() 

x1 = 1
y1 = 1 

result = divide(x1, y1) 
logger.info(f"result = {result}")

myfunc = RGBDecorators.conditional_decorator(condition = True, decorator = RGBDecorators.timeit)(divide)
result = myfunc(x1, y1)


#y2 = 0 
#
#result = divide(x1, y2) 
#logger.debug(f"result = {result}")





#
#
#    #def configure_logging(self, level = logging.INFO): 
#
#    #    #logging.basicConfig(level=level,
#    #    #            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s") 
#    #    #logger = get_logger() 
#
#    #    # Setting the login configuration from a dictionary 
#    #    logging.config.dictConfig() 
#
#
#    def get_logger(self, name): 
#        return logging.getLogger(name)




#def multiply(a = 1, b = 1):
#    return a * b 
#
#
#def add(a = 1, b = 1): 
#
#    return a + b
#
#
#if __name__ == "__main__":
#    # Configure logging from JSON or external file
#    #config_file = pb.Path("./logger_config.json")
#    #logger = configure_logging(config_file = config_file)
#    #print(logger)
#    #exit() 
#
#    ## Initialize the performance tracker with the configured logger
#    #tracker = PerformanceTracker(logger)
#
#    # Apply the performance tracker conditionally to the functions in the library
#    #tracked_some_function = tracker.time_and_track()(module1.some_function_in_module1)
#    #tracked_another_function = tracker.time_and_track()(module2.another_function_in_module2)
#
#    # Execute the functions with performance tracking
#    #tracked_some_function()
#    #tracked_another_function()
#
#    ## Call the functions, with logging and performance tracking based on configuration
#    #module1.some_function_in_module1()
#    #module2.another_function_in_module2()
#
#    a = 5
#    b = 10 
#
#    #result = tracker.time_and_track()(add)
#    #print(result(a, b))
#
#
#    # Configuration to be put inside json file 
#    LOGGING = {
#        "version": 1,
#        "disable_existing_loggers": False,
#        "formatters": {
#            "json": {
#                "format": "%(asctime)s %(levelname)s %(message)s",
#                "class": "pythonjsonlogger.jsonlogger.JsonFormatter",
#            }
#        },
#        "handlers": {
#            "stdout": {
#                "class": "logging.StreamHandler",
#                "stream": "ext://sys.stdout",
#                "formatter": "json",
#            }
#        },
#        "loggers": {"": {"handlers": ["stdout"], "level": "INFO"}},
#    }
#
#    
#    # We could have done it like this as well: 
#    # 
#    # logging.config.dictConfig(LOGGING)
#    # logger = logging.getLogger(__name__)
#    rgblogger = RGBLogger(config = LOGGING)
#
#    logger = rgblogger.get_logger(name = __name__) #logging.getLogger(__name__)
#
#    logger.info("An info")
#    logger.warning("A warning")







 
