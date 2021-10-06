"""Customized PyGeoUtils exceptions."""
from typing import Generator, List, Optional, Union


class EmptyResponse(Exception):
    """Exception raised when the input response is empty."""

    def __init__(self) -> None:
        self.message = "The input response is empty."
        super().__init__(self.message)

    def __str__(self) -> str:
        return self.message


class InvalidInputType(Exception):
    """Exception raised when a function argument type is invalid.

    Parameters
    ----------
    arg : str
        Name of the function argument
    valid_type : str
        The valid type of the argument
    example : str, optional
        An example of a valid form of the argument, defaults to None.
    """

    def __init__(self, arg: str, valid_type: str, example: Optional[str] = None) -> None:
        self.message = f"The {arg} argument should be of type {valid_type}"
        if example is not None:
            self.message += f":\n{example}"
        super().__init__(self.message)

    def __str__(self) -> str:
        return self.message


class InvalidInputValue(Exception):
    """Exception raised for invalid input.

    Parameters
    ----------
    inp : str
        Name of the input parameter
    valid_inputs : tuple
        List of valid inputs
    """

    def __init__(
        self, inp: str, valid_inputs: Union[List[str], Generator[str, None, None]]
    ) -> None:
        self.message = f"Given {inp} is invalid. Valid {inp}s are:\n" + ", ".join(
            str(i) for i in valid_inputs
        )
        super().__init__(self.message)

    def __str__(self) -> str:
        return self.message


class MissingAttribute(Exception):
    """Exception raised for missing attribute.

    Parameters
    ----------
    attr : str
        Name of the input attribute
    avail_attrs : tuple
        List of valid inputs
    """

    def __init__(self, attr: str, avail_attrs: Optional[List[str]] = None) -> None:
        if avail_attrs is not None:
            self.message = f"Given {attr} does not exist. Available attributes are:\n" + ", ".join(
                str(i) for i in avail_attrs
            )
        else:
            self.message = f"The {attr} attribute is missing."
        super().__init__(self.message)

    def __str__(self) -> str:
        return self.message


class UndeterminedDims(Exception):
    """Exception raised for dimensions cannot be determined."""

    def __init__(self) -> None:
        self.message = (
            "Dimensions cannot be determined from the dataset,"
            + "they must be provided via ds_dims argument."
        )
        super().__init__(self.message)

    def __str__(self) -> str:
        return self.message
