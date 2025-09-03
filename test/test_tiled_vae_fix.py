#!/usr/bin/env python3
"""
Test MPS compatibility for HYPIR Tiled VAE
"""

import torch
import sys
import os

# Add HYPIR to path
sys.path.append(os.path.join(os.path.dirname(__file__), "HYPIR"))

def test_custom_group_norm():
    """
    Test custom_group_norm with MPS device
    """
    from HYPIR.utils.tiled_vae.vaehook import custom_group_norm, get_var_mean
    
    # Check if MPS is available
    if not (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()):
        print("MPS not available, skipping test")
        return
    
    device = torch.device("mps")
    dtype = torch.float16
    
    print(f"Testing custom_group_norm on {device} with {dtype}")
    
    # Create test input
    batch_size, channels, height, width = 1, 32, 64, 64
    input_tensor = torch.randn(batch_size, channels, height, width, device=device, dtype=dtype)
    
    print(f"Input tensor: shape={input_tensor.shape}, device={input_tensor.device}, dtype={input_tensor.dtype}")
    
    # Calculate mean and var
    var, mean = get_var_mean(input_tensor, 32)
    print(f"Calculated mean: shape={mean.shape}, device={mean.device}, dtype={mean.dtype}")
    print(f"Calculated var: shape={var.shape}, device={var.device}, dtype={var.dtype}")
    
    # Test group norm
    try:
        result = custom_group_norm(input_tensor, 32, mean, var)
        print(f"✅ Success! Result: shape={result.shape}, device={result.device}, dtype={result.dtype}")
        print(f"Result stats: min={result.min():.4f}, max={result.max():.4f}, mean={result.mean():.4f}")
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_group_norm_param():
    """
    Test GroupNormParam.from_tile with MPS device
    """
    from HYPIR.utils.tiled_vae.vaehook import GroupNormParam
    
    # Check if MPS is available
    if not (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()):
        print("MPS not available, skipping test")
        return
    
    device = torch.device("mps")
    dtype = torch.float16
    
    print(f"\nTesting GroupNormParam.from_tile on {device} with {dtype}")
    
    # Create test tile
    batch_size, channels, height, width = 1, 32, 64, 64
    tile = torch.randn(batch_size, channels, height, width, device=device, dtype=dtype)
    
    # Create mock norm layer
    class MockNorm:
        def __init__(self, channels):
            self.weight = torch.ones(channels, device=device, dtype=dtype)
            self.bias = torch.zeros(channels, device=device, dtype=dtype)
    
    norm = MockNorm(channels)
    
    try:
        group_norm_func = GroupNormParam.from_tile(tile, norm)
        print(f"✅ GroupNormParam.from_tile created successfully")
        
        # Test the function
        test_input = torch.randn(batch_size, channels, height, width, device=device, dtype=dtype)
        result = group_norm_func(test_input)
        print(f"✅ Group norm function works! Result: shape={result.shape}, dtype={result.dtype}")
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=== Testing MPS Compatibility for HYPIR Tiled VAE ===")
    
    # Check PyTorch and MPS availability
    print(f"PyTorch version: {torch.__version__}")
    print(f"MPS available: {hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()}")
    
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        print(f"MPS built: {torch.backends.mps.is_built()}")
    
    print("\n" + "="*50)
    
    # Run tests
    test1_success = test_custom_group_norm()
    test2_success = test_group_norm_param()
    
    print("\n" + "="*50)
    print("Test Summary:")
    print(f"custom_group_norm test: {'✅ PASS' if test1_success else '❌ FAIL'}")
    print(f"GroupNormParam.from_tile test: {'✅ PASS' if test2_success else '❌ FAIL'}")
    
    if test1_success and test2_success:
        print("\n🎉 All tests passed! MPS compatibility should be working.")
    else:
        print("\n⚠️  Some tests failed. Check the error messages above.")