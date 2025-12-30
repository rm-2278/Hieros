"""Test args_type function with multiple entropy values."""

import sys
sys.path.insert(0, '/home/runner/work/Hieros/Hieros')

from hieros import tools


def test_args_type_single_string():
    """Test parsing single string value."""
    default = '3e-4'
    parser = tools.args_type(default)
    
    # Test single value
    result = parser('1e-3')
    assert result == '1e-3', f"Expected '1e-3', got {result}"
    print("✓ Single string value works")


def test_args_type_list_representation():
    """Test parsing string representation of a list."""
    default = '3e-4'
    parser = tools.args_type(default)
    
    # Test list representation (the problematic case)
    result = parser("['3e-4', '1e-3', '3e-3']")
    expected = ('3e-4', '1e-3', '3e-3')
    assert result == expected, f"Expected {expected}, got {result}"
    print("✓ List representation parsing works")


def test_args_type_comma_separated():
    """Test parsing comma-separated values."""
    default = ['3e-4']
    parser = tools.args_type(default)
    
    # Test comma-separated values
    result = parser('3e-4,1e-3,3e-3')
    expected = ('3e-4', '1e-3', '3e-3')
    assert result == expected, f"Expected {expected}, got {result}"
    print("✓ Comma-separated parsing works")


def test_schedule_with_parsed_list():
    """Test that schedule function works with parsed list values."""
    # Simulate what happens in hieros.py with get_layer_value
    default = '3e-4'
    parser = tools.args_type(default)
    
    # Parse the string representation
    parsed = parser("['3e-4', '1e-3', '3e-3']")
    
    # Simulate get_layer_value behavior
    for layer_idx in range(3):
        value = parsed[min(layer_idx, len(parsed) - 1)]
        # This should work without errors
        result = tools.schedule(value, step=100)
        assert isinstance(result, float), f"Expected float, got {type(result)}"
        print(f"  Layer {layer_idx}: schedule('{value}', 100) = {result}")
    
    print("✓ Schedule function works with parsed list values")


def test_edge_cases():
    """Test edge cases."""
    default = '3e-4'
    parser = tools.args_type(default)
    
    # Test with single element list
    result = parser("['3e-4']")
    expected = ('3e-4',)
    assert result == expected, f"Expected {expected}, got {result}"
    print("✓ Single element list works")
    
    # Test with different formats
    result = parser("['3e-3', '1e-3', '3e-4']")
    expected = ('3e-3', '1e-3', '3e-4')
    assert result == expected, f"Expected {expected}, got {result}"
    print("✓ Different order list works")


if __name__ == '__main__':
    print("Testing args_type function with multiple entropy values...\n")
    
    test_args_type_single_string()
    test_args_type_list_representation()
    test_args_type_comma_separated()
    test_schedule_with_parsed_list()
    test_edge_cases()
    
    print("\n✅ All tests passed!")
