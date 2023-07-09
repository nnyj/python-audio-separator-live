
def colorize(text, color):
    """
    Display text with some ANSI color in the terminal.
    """
    code = f"\033[{color}m"
    restore = "\033[0m"
    return "".join([code, text, restore])


def bold(text):
    """
    Display text in bold in the terminal.
    """
    return colorize(text, "1")
