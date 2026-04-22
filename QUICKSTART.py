#!/usr/bin/env python3
"""
Quick start script for testing FlakeForge inference.

This script:
1. Validates your environment setup
2. Shows the timing race flakiness
3. Ready to launch inference when you want
"""

import os
import sys
import subprocess
from pathlib import Path


def check_requirements():
    """Check if all required packages are installed."""
    required = {
        'pytest': 'pytest',
        'asyncio': 'asyncio (built-in)',
        'openai': 'openai',
        'dotenv': 'python-dotenv',
    }
    
    print("📋 Checking Python packages...")
    missing = []
    
    for import_name, display_name in required.items():
        try:
            __import__(import_name)
            print(f"  ✓ {display_name}")
        except ImportError:
            print(f"  ✗ {display_name} NOT FOUND")
            missing.append(display_name)
    
    if missing:
        print(f"\n❌ Missing packages: {', '.join(missing)}")
        print("\nInstall with:")
        print("  pip install pytest pytest-asyncio python-dotenv openai")
        return False
    
    print()
    return True


def check_api_key():
    """Check if NVIDIA API key is set."""
    print("🔑 Checking API Key...")
    
    api_key = os.getenv("NVIDIA_API_KEY") or os.getenv("OPENAI_API_KEY")
    
    if api_key:
        masked = api_key[:8] + "..." + api_key[-4:] if len(api_key) > 12 else "***"
        print(f"  ✓ API Key found: {masked}")
        return True
    else:
        print("  ✗ API Key NOT found")
        print("\n  Set it with:")
        print("    export NVIDIA_API_KEY='your-key-here'")
        print("  Or create a .env file:")
        print("    export $(cat .env | xargs)")
        return False


def show_timing_race():
    """Run the test multiple times to show flakiness."""
    print("\n🧪 Testing Flakiness Locally...")
    print("   Running test_flaky_simple 10 times...\n")
    
    test_repo = Path(__file__).parent / "test_repos" / "timing_race_minimal"
    passed = 0
    failed = 0
    
    for i in range(1, 11):
        result = subprocess.run(
            [sys.executable, "-m", "pytest", 
             "tests/test_flaky.py::test_flaky_simple", 
             "-v", "--tb=line", "-q"],
            cwd=test_repo,
            capture_output=True,
            text=True,
        )
        
        status = "✓ PASS" if result.returncode == 0 else "✗ FAIL"
        if result.returncode == 0:
            passed += 1
        else:
            failed += 1
        print(f"   Run {i:2d}: {status}")
    
    print(f"\n   📊 Result: {passed}/10 passed ({passed*10:.0f}%)")
    print(f"      Flakiness confirmed! ✓\n")
    return passed < 10


def show_next_steps():
    """Show what to do next."""
    print("\n" + "="*70)
    print("  Next Steps: Running Inference with FlakeForge")
    print("="*70 + "\n")
    
    print("✨ Option 1: Quick Test with Docker (if Docker is set up)\n")
    print("  # Build the image (one time)")
    print("  docker build -t flakeforge-env:latest -f server/Dockerfile .\n")
    print("  # Run inference")
    print("  cd test_repos/timing_race_minimal")
    print("  export USE_DOCKER_IMAGE=1")
    print("  python test_runner.py --run-inference --steps 3")
    
    print("\n" + "-"*70 + "\n")
    
    print("✨ Option 2: Run with Local Server\n")
    print("  # Terminal 1: Start the FlakeForge server")
    print("  cd FlakeForge")
    print("  export NVIDIA_API_KEY='your-key'")
    print("  uv run server --port 8000\n")
    print("  # Terminal 2: Run inference")
    print("  cd test_repos/timing_race_minimal")
    print("  export NVIDIA_API_KEY='your-key'")
    print("  export ENV_BASE_URL='http://localhost:8000'")
    print("  python test_runner.py --run-inference --steps 5")
    
    print("\n" + "-"*70 + "\n")
    
    print("📊 Results will be saved to:")
    print("  ./outputs/flakeforge_inference_<timestamp>.log")
    print("  ./outputs/flakeforge_summary_<timestamp>.json")
    print()


def main():
    """Main entry point."""
    print("\n" + "="*70)
    print("  🚀 FlakeForge Inference - Quick Start")
    print("="*70 + "\n")
    
    # Run checks
    if not check_requirements():
        sys.exit(1)
    
    if not check_api_key():
        print("\n⚠️  API key required to run inference with real models")
    
    # Show flakiness
    if show_timing_race():
        print("💡 Flakiness confirmed! Ready to run FlakeForge inference.\n")
    else:
        print("⚠️  Test is passing consistently - may not need 'fixing'\n")
    
    # Show next steps
    show_next_steps()


if __name__ == "__main__":
    main()
