import pathlib as pb 
import time 
import logging 
import json 
import tempfile 

import pytest 

from cimr_rgb.rgb_logging import RGBLogging # Assuming the class is saved in grid_generator.py



@pytest.fixture
def temp_log_dir():
    """
    Fixture to create a temporary directory for testing log files.

    This fixture:
    - Provides a temporary directory as a `pathlib.Path` object.
    - Automatically cleans up the directory after the test completes.

    Returns:
    -------
    pathlib.Path
        A temporary directory path for testing.
    """


    with tempfile.TemporaryDirectory() as tmp_dir:
        yield pb.Path(tmp_dir) 


@pytest.fixture
def sample_log_config():
    """
    Provides a sample logging configuration as a dictionary for testing.

    This fixture is used in tests that require a logging configuration.

    Returns:
    -------
    dict
        A dictionary representing a basic logging configuration.
    """

    return {
        "version": 1,
        "handlers": {
            "file": {
                "class": "logging.FileHandler",
                "filename": "test.log",
                "formatter": "default",
            }
        },
        "formatters": {
            "default": {
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            }
        },
        "root": {
            "level": "DEBUG",
            "handlers": ["file"],
        },
    }



@pytest.mark.parametrize("log_config_type", ["dict", "path"])
def test_rgb_logging_init(temp_log_dir, sample_log_config, log_config_type):
    """
    Unified test for initializing the `RGBLogging` class with a dictionary or a JSON file 
    log configuration.

    This test validates:
    - Proper initialization of the logging system when using a dictionary configuration.
    - Proper initialization when using a file-based JSON configuration.
    - Temporary JSON file creation and cleanup.

    Parameters:
    ----------
    temp_log_dir : pathlib.Path
        A temporary directory path provided by the `temp_log_dir` fixture.
    sample_log_config : dict
        A dictionary representing the logging configuration.
    log_config_type : str
        The type of logging configuration to test, either:
        - "dict": Use a dictionary for configuration.
        - "path": Write configuration to a JSON file and use the file path.

    Assertions:
    ----------
    - The logging system is initialized correctly for both types of configurations.
    - Temporary JSON file exists during the test for "path" configuration.
    """

    if log_config_type == "dict":
        # Test initialization with a dictionary log configuration
        rgb_logging = RGBLogging(logdir=temp_log_dir, log_config=sample_log_config)
        assert rgb_logging.log_config is None, "Log configuration should be initialized without errors."

    elif log_config_type == "path":
        # Write the log configuration to a temporary JSON file
        log_config_path = temp_log_dir / "log_config.json"
        #print(log_config_path)
        with open(log_config_path, "w") as f:
            json.dump(sample_log_config, f)

        # Confirm the file exists before the test ends
        assert log_config_path.exists(), "Temporary JSON file should exist during the test."

        # Test initialization with a file path log configuration
        rgb_logging = RGBLogging(logdir=temp_log_dir, log_config=log_config_path)
        assert rgb_logging.log_config is None, "Log configuration should load correctly from a JSON file."



def test_get_logger():
    """
    Test the `get_logger` method of the `RGBLogging` class.

    This test ensures that:
    - A logger is created with the correct name.
    - The logger object is returned successfully.

    Assertions:
    ----------
    - The logger's name matches the expected value.
    """

    rgb_logging = RGBLogging(logdir=pb.Path("/tmp"))
    
    logger = rgb_logging.get_logger("TestLogger")

    assert logger.name == "TestLogger", "Logger should be initialized with the correct name."




def test_rgb_decorated():
    """
    Test the `rgb_decorated` method of the `RGBLogging` class with a mock configuration 
    and function.

    This test validates:
    - The conditional decoration of a function based on configuration settings.
    - The behavior of the decorated function is preserved, and it returns the expected result.

    Assertions:
    ----------
    - The result of the decorated function matches the expected output.
    """

    class MockConfig:
        def __init__(self):
            self.logpar_decorate = True
            self.logger = logging.getLogger("MockLogger")

    def mock_function(a, b):
        return a + b

    config = MockConfig()
    decorated_func = RGBLogging.rgb_decorated(
        func=mock_function, rgb_config=config, decorator=RGBLogging.track_perf
    )
    result = decorated_func(2, 3)

    assert result == 5, "Decorated function should return the correct result."



def test_track_perf(caplog):
    """
    Tests the `track_perf` decorator from the `RGBLogging` class to ensure it correctly tracks 
    and logs performance metrics for a decorated function.

    This test validates the following:
    - The `track_perf` decorator does not alter the function's return value.
    - The decorator logs appropriate performance metrics, such as:
        - Function execution start.
        - Execution time.
        - Other CPU and memory-related metrics (if included in the logger output).
    - Logging is captured correctly using the `caplog` fixture.

    The test simulates a lightweight workload using the `test_function` function and asserts 
    that the logs contain specific messages to confirm the expected behavior of `track_perf`.

    Parameters:
    ----------
    caplog : pytest.LogCaptureFixture
        A Pytest fixture to capture log messages emitted during the test execution.

    Example:
    --------
    >>> def test_function():
    ...     sum(range(1000))  # Simulate workload
    ...     return "Completed"
    ...
    >>> logger = logging.getLogger("PerfLogger")
    >>> with caplog.at_level(logging.INFO):
    ...     tracked_result = RGBLogging.track_perf(func=test_function, logger=logger)
    ...     result = tracked_result()
    ...
    >>> assert result == "Completed"
    >>> assert any("Started Execution" in record.message for record in caplog.records)
    >>> assert any("Executed in:" in record.message for record in caplog.records)

    Asserts:
    -------
    - The result of the decorated function matches the original function's return value.
    - The logs contain specific messages indicating function execution start and timing metrics.
    """


    logger = logging.getLogger("PerfLogger")

    def test_function():
        sum(range(1000))  # Simulate workload
        return "Completed"

    with caplog.at_level(logging.INFO):
        tracked_result = RGBLogging.track_perf(func = test_function, logger = logger)
        result = tracked_result()

    assert result == "Completed", "Function should return the correct result."
    assert any("Started Execution" in record.message for record in caplog.records), "Should log 'Started Execution'."
    assert any("Executed in:" in record.message for record in caplog.records), "Should log execution time."



# Set up a basic logger for testing
# logging.basicConfig(level=logging.INFO)
# test_logger = logging.getLogger("TestLogger")


# @pytest.mark.parametrize("decorate, logger", [
#     (decorator, logger) for decorator in [True, False] for logger in [test_logger, None, logging.getLogger(__name__)]
# ])
# def test_rgb_decorated(decorate, logger):
#     """
#     Test the `rgb_decorated` method with various combinations of `decorate` and `logger`.

#     This test validates:
#     - If `decorate` is True and both `decorator` and `logger` are provided, the `rgb_decorated` method
#       should apply the decorator (`track_perf`) and preserve the function's behavior and output.
#     - If `decorate` is False or `logger` is None, the function should not be decorated and should
#       behave like a regular function.
#     - The function's output is preserved regardless of whether it is decorated or not.

#     Parameters:
#     - decorate: Boolean indicating whether to apply the decorator.
#     - logger: Logger instance to be passed to the decorator, or None.

#     The function `test_function` simulates a delay and returns a message based on the test parameters.
#     The output of the function is validated to ensure the decorator preserves the behavior and output.
#     """

#     message = f"decorate = {decorate}, logger = {logger}"

#     @RGBLogging.rgb_decorated(decorate=decorate, decorator=RGBLogging.track_perf, logger=logger)
#     def test_function():
#         time.sleep(0.1)

#         return message #f"Decorated!"

#     # Test execution
#     result = test_function()
#     assert result == message, "The decorator did not preserve the function's output."



# @pytest.fixture
# def rgb_logging_instance(temp_log_dir, sample_log_config):
#     """
#     Fixture to create an instance of RGBLogging for testing.
#     """
#     return RGBLogging(logdir=temp_log_dir, log_config=sample_log_config)



# @pytest.mark.parametrize("decorate, logger", [
#     (decorator, logger) for decorator in [True, False] for logger in [
#         logging.getLogger(__name__), 
#         test_logger 
#         ]
# ])
# def test_rgb_decorate_and_execute(decorate, logger, caplog):
def test_rgb_decorate_and_execute(caplog):
    """
    Test the `rgb_decorate_and_execute` method of the `RGBLogging` class with logging enabled.

    This test validates:
    - The method dynamically applies the specified decorator based on the provided configuration.
    - The decorated function logs performance metrics and returns the expected result.

    Parameters:
    ----------
    caplog : pytest.LogCaptureFixture
        A Pytest fixture to capture log messages emitted during the test execution.

    Example:
    --------
    >>> logger = logging.getLogger("DecorateLogger")
    >>> @RGBLogging.rgb_decorate_and_execute(decorate=True, decorator=RGBLogging.track_perf, logger=logger)
    >>> def test_function():
    ...     sum(range(1000))  # Simulate workload
    ...     return "Decorated"
    ...
    >>> with caplog.at_level(logging.INFO):
    ...     result = test_function()
    ...
    >>> assert result == "Decorated"
    >>> assert any("Started Execution" in record.message for record in caplog.records)

    Assertions:
    ----------
    - The decorated function returns the expected value.
    - Logs contain specific messages indicating function execution start and timing metrics.
    """

    logger = logging.getLogger("DecorateLogger")

    @RGBLogging.rgb_decorate_and_execute(
            decorate=True, #decorate, 
            decorator=RGBLogging.track_perf, 
            logger=logger
            )
    def test_function():
        sum(range(1000))  # Simulate workload
        return "Decorated"

    with caplog.at_level(logging.INFO):
        result = test_function()

    assert result == "Decorated", "Decorated function should return the correct value."
    assert any("Started Execution" in record.message for record in caplog.records), "Should log 'Started Execution'."

    # Assert logs are generated only if decorate=True and a valid logger is provided
    #if decorate:
    #    assert any("Started Execution" in record.message for record in caplog.records), \
    #        "Should log 'Started Execution' when decorated with logging."
    #else:
    #    assert len(caplog.records) == 0, "No logs should be recorded when decoration is disabled."

