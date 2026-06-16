"""Evolving Bias benchmark source package."""

import warnings


warnings.filterwarnings(
    "ignore",
    message=".*urllib3 .*charset_normalizer.*doesn't match a supported version.*",
)
warnings.filterwarnings(
    "ignore",
    message=".*urllib3 .*chardet.*doesn't match a supported version.*",
)
