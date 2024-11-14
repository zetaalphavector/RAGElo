"""A class for handling the CLI output with rich printing"""


class ConsoleWriter:
    def __init__(self, use_rich: bool = True):
        self.use_rich = use_rich

    def write(self, msg: str):
        print(msg)
