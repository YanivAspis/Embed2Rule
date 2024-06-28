import torch

def get_device():
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Using device {device}.")
    return device


ASP_NORMALISATION_MAP = {
    " ": "_SPACE_PLACEHOLDER_",
    ".": "_DOT_PLACEHOLDER_",
    "-": "_DASH_PLACEHOLDER_",
    "'": "_APOSTROPHE_PLACEHOLDER_",
    '"': "_DOUBLE_QUOTE_PLACEHOLDER_",
    "?": "_QUESTION_MARK_PLACEHOLDER_",
    "!": "_EXCLAMATION_MARK_PLACEHOLDER_",
    ":": "_COLON_PLACEHOLDER_",
    ";": "_SEMICOLON_PLACEHOLDER_",
    "/": "_FORWARD_SLASH_PLACEHOLDER_",
    "\\": "_BACKWARD_SLASH_PLACEHOLDER_",
    "[": "_OPEN_SQUARE_BRACKET_PLACEHOLDER_",
    "]": "_CLOSE_SQUARE_BRACKET_PLACEHOLDER_",
    "{": "_OPEN_CURLY_BRACKET_PLACEHOLDER_",
    "}": "_CLOSE_CURLY_BRACKET_PLACEHOLDER_",
    "<": "_OPEN_ANGLE_BRACKET_PLACEHOLDER_",
    ">": "_CLOSE_ANGLE_BRACKET_PLACEHOLDER_",
    "=": "_EQUALS_PLACEHOLDER_",
    "+": "_PLUS_PLACEHOLDER_",
    "*": "_STAR_PLACEHOLDER_",
    "&": "_AMPERSAND_PLACEHOLDER_",
    "%": "_PERCENT_PLACEHOLDER_",
    "$": "_DOLLAR_PLACEHOLDER_",
    "#": "_HASH_PLACEHOLDER_",
    "@": "_AT_PLACEHOLDER_",
    "^": "_CARET_PLACEHOLDER_",
    "|": "_PIPE_PLACEHOLDER_",
    "~": "_TILDE_PLACEHOLDER_",
    "`": "_GRAVE_ACCENT_PLACEHOLDER_",
}

def normalise_asp_constant(constant : str) -> str:
    for k, v in ASP_NORMALISATION_MAP.items():
        constant = constant.replace(k, v)
    return constant

def denormalise_asp_constant(constant : str) -> str:
    for k, v in ASP_NORMALISATION_MAP.items():
        constant = constant.replace(v, k)
    return constant
