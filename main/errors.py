# Custom Error Messages

class KeyErrorMessage(str):
    """
    https://stackoverflow.com/questions/46892261/
    """
    def __repr__(self): 
        return str(self)