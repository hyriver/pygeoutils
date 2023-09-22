"""Customized PyGeoUtils exceptions."""
from __future__ import annotations

from typing import Generator, Sequence


class MissingColumnError(Exception):
    """Exception raised when a required column is missing from a dataframe.

    Parameters
    ----------
    missing : list
        List of missing columns.
    """

    def __init__(self, missing: list[str]) -> None:
        self.message = "The following columns are missing:\n" + f"{', '.join(missing)}"
        super().__init__(self.message)

    def __str__(self) -> str:
        return self.message


class MissingCRSError(Exception):
    """Exception raised when CRS is not given."""

    def __init__(self) -> None:
        self.message = "CRS of the input geometry is missing."
        super().__init__(self.message)

    def __str__(self) -> str:
        return self.message


class MatchingCRSError(Exception):
    """Exception raised when CRS is not given."""

    def __init__(self) -> None:
        self.message = "Input dataframes must have the same CRS."
        super().__init__(self.message)

    def __str__(self) -> str:
        return self.message


class EmptyResponseError(Exception):
    """Exception raised when the input response is empty."""

    def __init__(self) -> None:
        self.message = "The input response is empty."
        super().__init__(self.message)

    def __str__(self) -> str:
        return self.message


class InputTypeError(TypeError):
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

    def __init__(self, arg: str, valid_type: str, example: str | None = None) -> None:
        self.message = f"The {arg} argument should be of type {valid_type}"
        if example is not None:
            self.message += f":\n{example}"
        super().__init__(self.message)

    def __str__(self) -> str:
        return self.message


class InputValueError(Exception):
    """Exception raised for invalid input.

    Parameters
    ----------
    inp : str
        Name of the input parameter
    valid_inputs : tuple
        List of valid inputs
    """

    def __init__(
        self, inp: str, valid_inputs: Sequence[str | int] | Generator[str | int, None, None]
    ) -> None:
        self.message = f"Given {inp} is invalid. Valid {inp}s are:\n" + ", ".join(
            str(i) for i in valid_inputs
        )
        super().__init__(self.message)

    def __str__(self) -> str:
        return self.message


class MissingAttributeError(Exception):
    """Exception raised for missing attribute.

    Parameters
    ----------
    attr : str
        Name of the input attribute
    avail_attrs : tuple
        List of valid inputs
    """

    def __init__(self, attr: str, avail_attrs: list[str] | None = None) -> None:
        if avail_attrs is not None:
            self.message = f"Given {attr} does not exist. Available attributes are:\n"
            self.message += ", ".join(avail_attrs)
        else:
            self.message = f"The {attr} attribute is missing."
        super().__init__(self.message)

    def __str__(self) -> str:
        return self.message


class InputRangeError(Exception):
    """Exception raised when a function argument is not in the valid range.

    Parameters
    ----------
    variable : str
        Variable with invalid value
    valid_range : str
        Valid range
    """

    def __init__(self, variable: str, valid_range: str) -> None:
        self.message = f"Valid range for {variable} is {valid_range}."
        super().__init__(self.message)

    def __str__(self) -> str:
        return self.message
