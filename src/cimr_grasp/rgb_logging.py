import pathlib as pb 
import time 
import json 
import logging.config   
import functools 
import psutil 





class RGBLogging:

    def __init__(self, log_config = None): 
        """

        STD docs for logging in python: 
        https://docs.python.org/3/howto/logging.html#configuring-logging
        https://docs.python.org/3/howto/logging.html#configuring-logging-for-a-library 

        Formatter attributes: 
        https://docs.python.org/3/library/logging.html#logrecord-attributes 
        """
        
        # log_file can be either dictionary, or json file  

        if log_config is not None: 

            if isinstance(log_config, pb.Path): 
                with open(log_config, "r") as file: 
                    log_config = json.load(file) 

            elif isinstance(log_config, str): 

                # Convert to Path to check if the string represents a file path
                log_config = pb.Path(log_config)
                with open(log_config, "r") as json_file: 
                    log_config = json.load(json_file) 

            elif isinstance(log_config, dict): 
                pass  
                print(log_config) 
            else: 
                raise TypeError("`log_config` can be str, Path or dictionary.")

            # Setting up RGB Logger configuration based on the provided file 
            self.log_config = logging.config.dictConfig(log_config) 

            # TODO: figure out this Queue handler and queue listener stuff 
            # Setting up the thread for Queue Handler 
            #queue_handler = logging.getHandlerByName("queue_handler")
            #if queue_handler is not None: 
            #    queue_handler.listener.start() 
            #    atexit.register(queue_handler.listener.stop) 

        else: 
            # TODO: make this into an error of sorts? 
            self.log_config = None 



    def get_logger(self, name):
        """
        Creates and returns a logger for a given module.
        This ensures consistent logging across the library.
        """

        logger = logging.getLogger(name)
        logger.debug(f"Getting logger named {name}")

        return logger 



    @staticmethod
    def rgb_decorated(decorate = False, decorator = None, logger = None):
    
        def wrapper(func): 
            if decorate and (decorator and logger) is not None: 
                return decorator(func, logger) 
            return func

            
        return wrapper
    


    @staticmethod 
    def track_perf(func, logger):

        @functools.wraps(func)
        def wrapper(*args, **kwargs):

            logger.info(f"Started execution of `{func.__name__}`")

            # Record the start time and resource usage
            start_time   = time.perf_counter()
            start_cpu    = psutil.cpu_percent(interval=None)
            start_memory = psutil.Process().memory_info().rss / (1024 * 1024)  # Memory in MB 

            
            result       = func(*args, **kwargs)

            # Record the end time and resource usage
            end_time     = time.perf_counter()
            end_cpu      = psutil.cpu_percent(interval=None)
            end_memory   = psutil.Process().memory_info().rss / (1024 * 1024)  # Memory in MB

            # Calculate time taken, CPU, and memory usage
            time_taken   = end_time - start_time
            cpu_usage    = end_cpu  - start_cpu
            memory_usage = end_memory - start_memory
    
            #logger.info(f"Finished execution of `{func.__name__}`: {time_taken:.2f}s")

            message = (f"`{func.__name__}` executed in {time_taken:.2f} seconds, "
               f"CPU usage: {cpu_usage:.2f}%, "
               f"Memory usage change: {memory_usage:.2f} MB.")

            logger.info(message) 
            
            return result
    
        return wrapper  
