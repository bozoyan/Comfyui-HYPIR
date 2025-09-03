#!/usr/bin/env python3
"""
Simplified test to check if SD2Enhancer can be imported correctly
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

def test_sd2_enhancer_import():
    """Test SD2Enhancer import using the same method as hypir_advanced_node.py"""
    print("Testing SD2Enhancer import...")
    
    # Import SD2Enhancer with better error handling (same as in hypir_advanced_node.py)
    SD2Enhancer = None
    try:
        from HYPIR.enhancer.sd2 import SD2Enhancer
        print("✓ Successfully imported SD2Enhancer via normal import")
        return True
    except ImportError as e:
        print(f"❌ Normal import failed: {e}")
        print("Trying alternative import...")
        try:
            # Try direct import from sd2 module
            import importlib.util
            sd2_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "HYPIR", "enhancer", "sd2.py")
            if os.path.exists(sd2_path):
                spec = importlib.util.spec_from_file_location("sd2", sd2_path)
                if spec is not None and spec.loader is not None:
                    sd2_module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(sd2_module)
                    SD2Enhancer = sd2_module.SD2Enhancer
                    print("✓ Successfully imported SD2Enhancer via direct module loading")
                    return True
                else:
                    print(f"❌ Could not create module spec for {sd2_path}")
            else:
                print(f"❌ sd2.py not found at {sd2_path}")
        except Exception as e2:
            print(f"❌ Alternative import also failed: {e2}")
            return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

if __name__ == "__main__":
    success = test_sd2_enhancer_import()
    sys.exit(0 if success else 1)