import warnings


warnings.filterwarnings(
    "ignore",
    message="urllib3 .*doesn't match a supported version!",
    category=Warning,
)

