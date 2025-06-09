# utils/async_resilience_v2_5.py
# EOTS v2.5 - SENTRY-APPROVED RESILIENCE PATTERN

import asyncio
import logging
from functools import wraps
from typing import Callable, Any
import aiohttp

logger = logging.getLogger(__name__)

def async_retry(max_attempts: int = 3, backoff_factor: float = 0.5) -> Callable:
    """
    An asynchronous retry decorator for resilient aiohttp requests.

    This decorator wraps an asynchronous function and retries it upon encountering
    aiohttp.ClientError exceptions, using an exponential backoff strategy.

    Args:
        max_attempts (int): The maximum number of attempts to make.
        backoff_factor (float): The factor to apply for exponential backoff delay.

    Returns:
        Callable: The wrapped asynchronous function.
    
    Justification:
        This directly implements the Asynchronous Resilience Decorator pattern
        from the S-Grade blueprint, adapted from the patterns in the
        Python_Expert_Patterns_Toolkit.md.
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            attempts = 0
            last_exception = None
            while attempts < max_attempts:
                attempts += 1
                try:
                    return await func(*args, **kwargs)
                except aiohttp.ClientError as e:
                    last_exception = e
                    if attempts >= max_attempts:
                        logger.error(f"Function {func.__name__} failed after {max_attempts} attempts. Final error: {e}")
                        break 
                    
                    delay = backoff_factor * (2 ** (attempts - 1))
                    logger.warning(
                        f"Attempt {attempts}/{max_attempts} for {func.__name__} failed with {type(e).__name__}. "
                        f"Retrying in {delay:.2f} seconds."
                    )
                    await asyncio.sleep(delay)
            
            raise last_exception
        return wrapper
    return decorator