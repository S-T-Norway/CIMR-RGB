import sys 
import pathlib as pb 
import time, datetime  
import json 
import logging.config   
import functools 
import typing 

import psutil 


class RGBLogging:
    """
    A utility class for configuring and managing logging in applications, 
    with support for performance tracking and dynamic function decoration.

    Features:
    ----------
    - Allows logging configuration using a dictionary or JSON file.
    - Provides a consistent interface for creating and retrieving loggers.
    - Includes a utility for dynamically decorating functions with a performance-tracking decorator.

    Methods:
    ----------
    - __init__(logdir, log_config): Initializes the logging configuration.
    - get_logger(name): Creates and returns a logger.
    - rgb_decorate_and_execute(decorate, decorator, logger): Dynamically decorates a function based on input parameters.
    - track_perf(func, logger): Decorates a function to log its performance metrics.

    Example Usage:
    ----------
    >>> rgb_logging = RGBLogging(logdir=Path("/logs"), log_config={"handlers": {...}})
    >>> logger = rgb_logging.get_logger("MyLogger")
    >>> @RGBLogging.track_perf(func=my_function, logger=logger)
    >>> def my_function():
    >>>     pass
    """

    def __init__(self, logdir: pb.Path, 
                 log_config = None, 
                 filename = "cimr", 
                 #name_prefix = "cimr", 
                 #name_suffix = f"{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')}", 
                 file_extension = ".log"
                 ) -> None: 

        """
        Initializes the logging configuration.

        Parameters:
        ----------
        logdir : pathlib.Path
            The directory where log files will be stored.
        log_config : Union[dict, pathlib.Path, str, None], optional
            The logging configuration. Can be provided as:
            - A dictionary.
            - A JSON file path as a string or `pathlib.Path`.
            - None (default) if no configuration is provided.

        Raises:
        ----------
        TypeError:
            If `log_config` is not a dictionary, string, or `pathlib.Path`.

        Behavior:
        ----------
        - If a dictionary is provided, it is directly used as the logging configuration.
        - If a JSON file path is provided, the file is loaded and used as the configuration.
        - The log file name is modified to include a timestamp.

        Example:
        ----------
        >>> logdir = pathlib.Path("/logs")
        >>> log_config = {
        >>>     "version": 1,
        >>>     "handlers": {
        >>>         "file": {"class": "logging.FileHandler", "filename": "test.log"}
        >>>     }
        >>> }
        >>> rgb_logging = RGBLogging(logdir, log_config)
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
                #print(log_config) 
                pass  
            else: 
                raise TypeError("`log_config` can be str, Path or dictionary.")

            #print(log_config['handlers']['file']['filename'])
            # Getting the name of log file 
            #modified_config_name = pb.Path(log_config['handlers']['file']['filename']).name  
            # Modifying it to include time signature 
            modified_config_name = filename + file_extension 
            #modified_config_name = name_prefix + \
            #                    f"_{name_suffix}" + f"{file_extension}" 
            #                    f"_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')}" + ".log" 
            # Creating absolute path for the log config file 
            modified_config_name = logdir.joinpath(modified_config_name)

            log_config['handlers']['file']['filename'] = modified_config_name 
            
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



    def get_logger(self, name: str):
        """
        Creates and returns a logger for the given module or component.

        Parameters:
        ----------
        name : str
            The name of the logger, typically the module or component name.

        Returns:
        ----------
        logging.Logger:
            A configured logger instance.

        Example:
        ----------
        >>> rgb_logging = RGBLogging(logdir=pathlib.Path("/logs"))
        >>> logger = rgb_logging.get_logger("MyLogger")
        >>> logger.info("This is a log message.")
        """

        logger = logging.getLogger(name)
        logger.debug(f"Getting logger named {name}")

        return logger 



    @staticmethod
    #def rgb_decorated(decorate: bool = False, 
    #                  decorator: typing.Optional[typing.Callable] = None, 
    #                  logger: typing.Optional[logging.Logger] = None
    #                  ):  
    def rgb_decorate_and_execute(decorate: bool = False, 
                      decorator: typing.Optional[typing.Callable] = None, 
                      logger: typing.Optional[logging.Logger] = None
                      ):  

        """
        Conditionally decorates a function with the provided decorator.

        Parameters:
        ----------
        decorate : bool, optional
            Whether to apply the decorator (default: False).
        decorator : Callable, optional
            The decorator function to apply (default: None).
        logger : logging.Logger, optional
            The logger instance to pass to the decorator, if applicable (default: None).

        Returns:
        ----------
        Callable:
            A wrapper function that applies the decorator conditionally.

        Example:
        ----------
        >>> logger = logging.getLogger("MyLogger")
        >>> @RGBLogging.rgb_decorate_and_execute(decorate=True, decorator=RGBLogging.track_perf, logger=logger)
        >>> def my_function():
        >>>     pass
        """

        def outer_wrapper(func: typing.Callable): 
            if decorate and decorator is not None and logger is not None: 

                #logger.info("---------------------")
                logger.info(f"`{func.__name__}`")

                return decorator(func, logger) 
            return func

            
        return outer_wrapper
    


    @staticmethod 
    def track_perf(func: typing.Callable, logger: logging.Logger):
        """
        Decorates a function to log its performance metrics, including execution time, CPU usage, and memory usage.

        Parameters:
        ----------
        func : Callable
            The target function to be decorated.
        logger : logging.Logger
            The logger instance to use for logging performance metrics.

        Returns:
        ----------
        Callable:
            The decorated function.

        Logged Metrics:
        ----------
        - Start of execution.
        - Execution time in seconds.
        - CPU user and system time.
        - Memory usage change during execution.

        Example:
        ----------
        >>> logger = logging.getLogger("PerfLogger")
        >>> @RGBLogging.track_perf(func=my_function, logger=logger)
        >>> def my_function():
        >>>     pass
        """

        @functools.wraps(func)
        def perf_wrapper(*args, **kwargs):

            logger.info("---------------------")

            logger.info(f"`{func.__name__}` -- Started Execution") 

            # Get the current process 
            process = psutil.Process() 

            # Record the start time and resource usage
            start_time   = time.perf_counter()

            cpu_time_start = process.cpu_times() 
            cpu_usage_start = process.cpu_percent(interval=None) 

            # Memory size in bytes / 1024**2 = memory size in MB 
            memory_usage_start = process.memory_info().rss / 1024**2 

            # Execute the function 
            result       = func(*args, **kwargs)

            # Record the end time and resource usage
            end_time     = time.perf_counter()

            cpu_time_end  = process.cpu_times() 
            cpu_usage_end = process.cpu_percent(interval=None) 

            # Memory size in bytes / 1024**2 = memory size in MB 
            memory_usage_end = process.memory_info().rss / 1024**2 

            # Calculate metrics 
            elapsed_time   = end_time - start_time

            user_cpu_time = cpu_time_end.user - cpu_time_start.user 
            system_cpu_time = cpu_time_end.system - cpu_time_start.system 
            total_cpu_time  = user_cpu_time + system_cpu_time 

            memory_usage_change = memory_usage_end - memory_usage_start 

            # Log performance metrics 
            logger.info(f"`{func.__name__}` -- Executed in: {elapsed_time:.2f}s") 
            logger.info(f"`{func.__name__}` -- CPU User Time (Change): {user_cpu_time:.2f}s") 
            logger.info(f"`{func.__name__}` -- CPU System Time: {system_cpu_time:.2f}s") 
            logger.info(f"`{func.__name__}` -- CPU Total Time: {total_cpu_time:.2f}s") 
            logger.info(f"`{func.__name__}` -- Process-Specific CPU Usage (Before): {cpu_usage_start:.2f}%") 
            logger.info(f"`{func.__name__}` -- Process-Specific CPU Usage (After): {cpu_usage_end:.2f}%") 
            logger.info(f"`{func.__name__}` -- Memory Usage Change: {memory_usage_change:.6f} MB") 

            logger.info("---------------------")
            
            
            return result

        return perf_wrapper  
    

    @staticmethod 
    def handle_error(error: Exception, message: str, logger: logging.Logger, level="error"):
        """
        Centralized method for handling and logging errors.

        Parameters:
        ----------
        error : Exception
            The exception instance.
        message : str
            A custom message to log alongside the exception.
        level : str, optional
            The logging level: "error", "warning", or "critical" (default: "error").
        """

        if level == "error":

            logger.error(f"{error}: {message}", exc_info=True)

        elif level == "warning":

            logger.warning(f"{error}: {message}")

        elif level == "critical":

            logger.critical(f"{error}: {message}")

        else:
            logger.info(f"{error}: {message}")





    def setup_global_exception_handler(self, logger):
        """
        Redirect uncaught exceptions to the logger.
        """

        def handle_exception(exc_type, exc_value, exc_traceback):

            if issubclass(exc_type, KeyboardInterrupt):

                sys.__excepthook__(exc_type, exc_value, exc_traceback)

                return

            logger.critical(
                "Uncaught exception",
                exc_info=(exc_type, exc_value, exc_traceback),
            )

        sys.excepthook = handle_exception



    # TODO: This function should simplify the call for decorator, but is really not needed. 
    # @staticmethod  
    # def rgb_decorated(func, rgb_config, decorator):
    #     """
    #     Dynamically applies a specified decorator to a given function based on configuration settings.

    #     This method uses the `rgb_decorate_and_execute` decorator factory to conditionally apply the provided 
    #     decorator to the target function. The decision to decorate is controlled by the `logpar_decorate` 
    #     attribute of the `rgb_config` object. If `logpar_decorate` is True and a valid decorator is provided, 
    #     the function is wrapped with the specified decorator.

    #     Parameters:
    #     ----------
    #     func : Callable
    #         The target function to be decorated.
    #     
    #     rgb_config : object
    #         A configuration object that must have the following attributes:
    #         - `logpar_decorate` (bool): Determines whether the function should be decorated.
    #         - `logger` (logging.Logger): A logger instance to pass to the decorator.
    #     
    #     decorator : Callable
    #         The decorator function to apply to `func`. This decorator must accept the target 
    #         function and the logger as arguments.

    #     Returns:
    #     -------
    #     Callable
    #         The decorated function if `logpar_decorate` is True, otherwise the original function.

    #     Example:
    #     -------
    #     >>> # Example function
    #     >>> def add(a, b):
    #     ...     return a + b
    #     ...
    #     >>> # Configuration class
    #     >>> class RGBConfig:
    #     ...     def __init__(self):
    #     ...         self.logpar_decorate = True
    #     ...         self.logger = logging.getLogger("ExampleLogger")
    #     ...
    #     >>> rgb_config = RGBConfig()
    #     >>> decorated_add = rgb_decorate_and_execute(add, rgb_config, RGBLogging.track_perf)
    #     >>> result = decorated_add(4, 5)
    #     >>> print(result)
    #     9
    #     """
    #     decorated_func = RGBLogging.rgb_decorate_and_execute(
    #         decorate=rgb_config.logpar_decorate,
    #         decorator=decorator,
    #         logger=rgb_config.logger,
    #     )(func)
    #     return decorated_func


