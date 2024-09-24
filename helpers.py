# Custom exception for invalid input to the Generalized Black-Scholes (GBS) function
class GBS_InputError(Exception):
    def __init__(self, error_message):
        super().__init__(error_message)

# Custom exception for calculation errors in the GBS function
class GBS_CalculationError(Exception):
    def __init__(self, error_message):
        super().__init__(error_message)

def assert_close(value1, value2, tolerance=0.000001):
    """
    Test if two floating-point numbers are approximately equal.
    
    For numbers less than 1 million, they are considered equal if their absolute difference
    is within the tolerance. For larger numbers, they are considered equal if their
    relative difference is within 0.0001%.
    
    Args:
    value1 (float): First value to compare
    value2 (float): Second value to compare
    tolerance (float): Tolerance for equality check (default: 0.000001)
    
    Returns:
    bool: True if the numbers are considered equal, False otherwise
    """
    if (value1 < 1000000.0) and (value2 < 1000000.0):
        difference = abs(value1 - value2)
        difference_type = "Absolute Difference"
    else:
        difference = abs((value1 - value2) / value1)
        difference_type = "Relative Difference"

    is_close = difference < tolerance

    if (__name__ == "__main__") and (not is_close):
        print(f"  FAILED TEST. Comparing {value1} and {value2}. "
              f"{difference_type} is {difference}, Tolerance is {tolerance}")
    else:
        print(".")

    return is_close