import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)

log = logging.getLogger("vla_network")

def warn_when(condition, message):
    """Log a warning when the condition is met."""
    if condition:
        log.warning(message)

log.warn_when = warn_when
