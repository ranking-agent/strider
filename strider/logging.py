import logging
from .storage import RedisList, r
from enum import Enum


class LogLevelEnum(str, Enum):
    """
    Python logging module log level constants represented as an ``enum.Enum``.
    """
    NOTSET = 'NOTSET'
    DEBUG = 'DEBUG'
    INFO = 'INFO'
    WARNING = 'WARNING'
    ERROR = 'ERROR'
    CRITICAL = 'CRITICAL'
