from collections import deque
import time

from ..utils.logger_utils import get_logger

logger = get_logger(__name__)

class SlidingWindowRateLimiter:
    def __init__(self, max_requests, period):
        self.max_requests = max_requests
        self.period = period
        self.window = deque()
        self.last_request_time = 0
        self.backoff_time = 1  # Initial backoff time in seconds
        logger.info(f"Rate limiter initialized with max_requests: {max_requests}, period: {period}")

    def _clean_window(self, current_time):
        while self.window and current_time - self.window[0] > self.period:
            self.window.popleft()

    def wait_for_token(self):
        current_time = time.time()
        self._clean_window(current_time)

        if len(self.window) < self.max_requests:
            wait_time = 0
        else:
            wait_time = max(0, self.period - (current_time - self.window[0]))

        if self.window:
            time_since_last_request = current_time - self.last_request_time
            ideal_spacing = self.period / self.max_requests
            if time_since_last_request < ideal_spacing:
                wait_time = max(wait_time, ideal_spacing - time_since_last_request)

        if wait_time > 0:
            time.sleep(wait_time)

        current_time = time.time()
        self.window.append(current_time)
        self.last_request_time = current_time

        # Reset backoff time after a successful request
        self.backoff_time = 1

    def backoff(self):
        logger.debug(f"Backing off for {self.backoff_time} seconds")
        time.sleep(self.backoff_time)
        self.backoff_time = min(self.backoff_time * 2, 60)