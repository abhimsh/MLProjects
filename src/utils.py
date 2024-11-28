import dill
import os

from src.logger import logging
from src.exception_handler import UtilsException


def save_object(file_path: str, obj) -> None:
    """
    Will save the python object into file using the pickling
    """
    try:
        logging.debug(f"({file_path}, {repr(obj)})")
        # Create the folder if not present
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # Write the object to file
        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

        logging.debug(f"Successfully written {repr(file_obj)} into {file_path}")

    except Exception as error:
        logging.exception("Error Encounterd while writing obj to file")
        raise UtilsException(error)