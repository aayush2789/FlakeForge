"""Financial calculations with floating-point arithmetic."""

def total_with_tax(prices: list, tax_rate: float = 0.1) -> float:
    """Sum prices and add tax. Bug: floating-point accumulation error."""
    subtotal = 0.0
    for p in prices:
        subtotal += p
    return subtotal * (1 + tax_rate)

def split_bill(total: float, people: int) -> float:
    """Split total evenly. Bug: float division may not be exact."""
    return total / people

def verify_split(total: float, people: int) -> bool:
    """Verify that split * people == total. Bug: uses == on floats."""
    per_person = split_bill(total, people)
    return per_person * people == total
