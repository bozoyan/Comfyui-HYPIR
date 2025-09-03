#!/usr/bin/env python3
"""
Test script to verify HYPIR imports are working correctly.
"""
import sys
import os

# Add current directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# Add HYPIR directory to path
hypir_path = os.path.join(current_dir, "HYPIR")
if hypir_path not in sys.path:
    sys.path.append(hypir_path)

def test_imports():
    """Test all critical imports for HYPIR plugin"""
    print("Testing HYPIR imports...")
    
    try:
        print("1. Testing basic Python modules...")
        import torch
        import numpy as np
        from PIL import Image
        print("   ✓ Basic modules imported successfully")
        
        print("2. Testing HYPIR utils imports...")
        from HYPIR.utils.common import wavelet_reconstruction, make_tiled_fn
        from HYPIR.utils.tiled_vae import enable_tiled_vae
        print("   ✓ HYPIR utils imported successfully")
        
        print("3. Testing HYPIR enhancer imports...")
        from HYPIR.enhancer.base import BaseEnhancer
        print("   ✓ BaseEnhancer imported successfully")
        
        from HYPIR.enhancer.sd2 import SD2Enhancer
        print("   ✓ SD2Enhancer imported successfully")
        
        print("4. Testing node imports...")
        from hypir_config import HYPIR_CONFIG
        from model_downloader import get_hypir_model_path
        print("   ✓ Config and downloader imported successfully")
        
        print("5. Testing advanced node import...")
        from hypir_advanced_node import HYPIRAdvancedRestoration
        print("   ✓ HYPIRAdvancedRestoration imported successfully")
        
        print("\\n🎉 All imports successful! HYPIR should work correctly now.")
        return True
        
    except ImportError as e:
        print(f"\\n❌ Import error: {e}")
        import traceback
        traceback.print_exc()
        return False
    except Exception as e:
        print(f"\\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_imports()
    sys.exit(0 if success else 1)