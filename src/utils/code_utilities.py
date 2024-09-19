import timeit
from typing import Any, Callable, Optional
import logging

def timing_decorator(logger: Optional[logging.Logger] = None):
    """
    A decorator factory that creates a decorator which measures the execution time of a function and logs or prints the duration.

    Parameters:
        logger (Optional[logging.Logger], optional): The logger to be used for logging the duration. Defaults to None.

    Returns:
        Callable[..., Any]: The decorator.
    """

    def decorator(func: Callable[..., Any]):
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            start = timeit.default_timer()
            result = func(*args, **kwargs)
            end = timeit.default_timer()
            if logger:
                logger.info(f"{func.__name__} took {end - start} seconds to run")
            else:
                print(f"{func.__name__} took {end - start} seconds to run")
            return result

        return wrapper

    return decorator