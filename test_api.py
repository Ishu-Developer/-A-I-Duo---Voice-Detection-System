import requests
import json
import os
import sys
from pathlib import Path
from typing import Dict, Any

# ============================================
# CONFIGURATION
# ============================================
API_URL = "https://a-i-duo-voice-detection-system-production.up.railway.app"
API_KEY = "voice_test_2026"
SUPPORTED_LANGUAGES = ["Tamil", "English", "Hindi", "Malayalam", "Telugu"]
TIMEOUT = 30

# Color codes for console output
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    END = '\033[0m'
    BOLD = '\033[1m'

# ============================================
# UTILITY FUNCTIONS
# ============================================

def print_header(text: str) -> None:
    """Print formatted header"""
    print(f"\n{Colors.BOLD}{Colors.CYAN}{'='*60}")
    print(f"{text.center(60)}")
    print(f"{'='*60}{Colors.END}\n")

def print_success(text: str) -> None:
    """Print success message"""
    print(f"{Colors.GREEN}✓ {text}{Colors.END}")

def print_error(text: str) -> None:
    """Print error message"""
    print(f"{Colors.RED}✗ {text}{Colors.END}")

def print_info(text: str) -> None:
    """Print info message"""
    print(f"{Colors.BLUE}ℹ {text}{Colors.END}")

def print_warning(text: str) -> None:
    """Print warning message"""
    print(f"{Colors.YELLOW}⚠ {text}{Colors.END}")

# ============================================
# TEST FUNCTIONS
# ============================================

def test_health_check() -> bool:
    """Test 1: Health Check Endpoint"""
    print_header("Test 1: Health Check")
    
    try:
        response = requests.get(
            f"{API_URL}/health",
            timeout=TIMEOUT
        )
        
        if response.status_code == 200:
            data = response.json()
            print_success("Health check endpoint working")
            print(json.dumps(data, indent=2))
            
            # Verify response structure
            required_fields = ["status", "model", "model_loaded", "scaler", "supported_languages"]
            for field in required_fields:
                if field in data:
                    print_success(f"Field '{field}' present")
                else:
                    print_error(f"Field '{field}' missing")
                    return False
            
            return True
        else:
            print_error(f"Unexpected status code: {response.status_code}")
            return False
    
    except Exception as e:
        print_error(f"Connection error: {str(e)}")
        return False

def test_root_endpoint() -> bool:
    """Test 2: Root Endpoint"""
    print_header("Test 2: Root Endpoint")
    
    try:
        response = requests.get(
            f"{API_URL}/",
            timeout=TIMEOUT
        )
        
        if response.status_code == 200:
            data = response.json()
            print_success("Root endpoint working")
            print(json.dumps(data, indent=2))
            
            if "message" in data and "endpoints" in data:
                print_success("Response structure valid")
                return True
            else:
                print_error("Invalid response structure")
                return False
        else:
            print_error(f"Unexpected status code: {response.status_code}")
            return False
    
    except Exception as e:
        print_error(f"Connection error: {str(e)}")
        return False

def test_invalid_api_key() -> bool:
    """Test 3: Invalid API Key Handling"""
    print_header("Test 3: Invalid API Key Error Handling")
    
    try:
        headers = {"x-api-key": "invalid_key_12345"}
        response = requests.post(
            f"{API_URL}/api/voice-detection-file?language=English",
            headers=headers,
            files={"file": ("test.mp3", b"fake audio data")},
            timeout=TIMEOUT
        )
        
        if response.status_code == 403:
            print_success("Invalid API key correctly rejected (403)")
            print(f"Response: {response.json()}")
            return True
        else:
            print_error(f"Expected 403, got {response.status_code}")
            return False
    
    except Exception as e:
        print_error(f"Test failed: {str(e)}")
        return False

def test_invalid_language() -> bool:
    """Test 4: Invalid Language Handling"""
    print_header("Test 4: Invalid Language Error Handling")
    
    try:
        headers = {"x-api-key": API_KEY}
        response = requests.post(
            f"{API_URL}/api/voice-detection-file?language=InvalidLang",
            headers=headers,
            files={"file": ("test.mp3", b"fake audio data")},
            timeout=TIMEOUT
        )
        
        if response.status_code == 400:
            print_success("Invalid language correctly rejected (400)")
            print(f"Response: {response.json()}")
            return True
        else:
            print_error(f"Expected 400, got {response.status_code}")
            return False
    
    except Exception as e:
        print_error(f"Test failed: {str(e)}")
        return False

def test_file_upload(file_path: str, language: str = "English") -> Dict[str, Any]:
    """Test 5: File Upload with Real Audio"""
    print_header(f"Test 5: File Upload - {language}")
    
    if not os.path.exists(file_path):
        print_warning(f"File not found: {file_path}")
        print_info("Skipping file upload test")
        return None
    
    try:
        headers = {"x-api-key": API_KEY}
        file_size = os.path.getsize(file_path)
        
        print_info(f"File: {os.path.basename(file_path)}")
        print_info(f"Size: {file_size / (1024*1024):.2f} MB")
        print_info(f"Language: {language}")
        
        with open(file_path, 'rb') as f:
            response = requests.post(
                f"{API_URL}/api/voice-detection-file",
                params={"language": language},
                headers=headers,
                files={"file": f},
                timeout=TIMEOUT
            )
        
        if response.status_code == 200:
            data = response.json()
            print_success("File upload successful")
            
            # Print results
            print(f"\n{Colors.BOLD}Results:{Colors.END}")
            print(f"  Status: {data.get('status')}")
            print(f"  Prediction: {Colors.BOLD}{data.get('prediction')}{Colors.END}")
            print(f"  Confidence: {data.get('confidence'):.2%}")
            print(f"  Processing Time: {data.get('processing_time_ms'):.2f}ms")
            print(f"  Request ID: {data.get('request_id')}")
            
            return data
        
        elif response.status_code == 413:
            print_error("File too large (max 10MB)")
            return None
        
        elif response.status_code == 400:
            print_error(f"Bad request: {response.json()}")
            return None
        
        else:
            print_error(f"Unexpected status code: {response.status_code}")
            print_error(f"Response: {response.text}")
            return None
    
    except Exception as e:
        print_error(f"Upload failed: {str(e)}")
        return None

def test_base64_encoding(file_path: str, language: str = "English") -> Dict[str, Any]:
    """Test 6: Base64 Encoded Audio"""
    print_header(f"Test 6: Base64 Encoding - {language}")
    
    if not os.path.exists(file_path):
        print_warning(f"File not found: {file_path}")
        print_info("Skipping Base64 test")
        return None
    
    try:
        import base64
        
        # Read and encode file
        with open(file_path, 'rb') as f:
            audio_bytes = f.read()
        
        audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
        
        print_info(f"File: {os.path.basename(file_path)}")
        print_info(f"Original size: {len(audio_bytes)} bytes")
        print_info(f"Base64 encoded size: {len(audio_base64)} bytes")
        
        # Prepare request
        payload = {
            "language": language,
            "audioFormat": Path(file_path).suffix[1:],  # Remove leading dot
            "audioBase64": audio_base64
        }
        
        headers = {
            "x-api-key": API_KEY,
            "Content-Type": "application/json"
        }
        
        response = requests.post(
            f"{API_URL}/api/voice-detection",
            json=payload,
            headers=headers,
            timeout=TIMEOUT
        )
        
        if response.status_code == 200:
            data = response.json()
            print_success("Base64 detection successful")
            
            print(f"\n{Colors.BOLD}Results:{Colors.END}")
            print(f"  Status: {data.get('status')}")
            print(f"  Prediction: {Colors.BOLD}{data.get('prediction')}{Colors.END}")
            print(f"  Confidence: {data.get('confidence'):.2%}")
            print(f"  Processing Time: {data.get('processing_time_ms'):.2f}ms")
            
            return data
        
        else:
            print_error(f"Request failed: {response.status_code}")
            print_error(f"Response: {response.json()}")
            return None
    
    except Exception as e:
        print_error(f"Base64 test failed: {str(e)}")
        return None

def test_multiple_languages(file_path: str) -> bool:
    """Test 7: Test All Supported Languages"""
    print_header("Test 7: Multi-language Support")
    
    if not os.path.exists(file_path):
        print_warning(f"File not found: {file_path}")
        return False
    
    results = {}
    
    for language in SUPPORTED_LANGUAGES:
        try:
            print_info(f"Testing {language}...")
            
            headers = {"x-api-key": API_KEY}
            
            with open(file_path, 'rb') as f:
                response = requests.post(
                    f"{API_URL}/api/voice-detection-file",
                    params={"language": language},
                    headers=headers,
                    files={"file": f},
                    timeout=TIMEOUT
                )
            
            if response.status_code == 200:
                data = response.json()
                results[language] = {
                    "prediction": data.get('prediction'),
                    "confidence": data.get('confidence')
                }
                print_success(f"{language}: {data.get('prediction')} ({data.get('confidence'):.2%})")
            else:
                print_error(f"{language}: Failed with status {response.status_code}")
                results[language] = {"error": response.status_code}
        
        except Exception as e:
            print_error(f"{language}: {str(e)}")
            results[language] = {"error": str(e)}
    
    print(f"\n{Colors.BOLD}Summary:{Colors.END}")
    print(json.dumps(results, indent=2))
    
    return len(results) == len(SUPPORTED_LANGUAGES)

def test_case_insensitive_language() -> bool:
    """Test 8: Case-Insensitive Language Support"""
    print_header("Test 8: Case-Insensitive Language Handling")
    
    test_cases = [
        ("english", "ENGLISH"),
        ("tamil", "TAMIL"),
        ("hindi", "HINDI"),
    ]
    
    for lower_case, upper_case in test_cases:
        try:
            headers = {"x-api-key": API_KEY}
            response = requests.post(
                f"{API_URL}/api/voice-detection-file?language={lower_case}",
                headers=headers,
                files={"file": ("test.mp3", b"fake audio data")},
                timeout=TIMEOUT
            )
            
            # We expect a file-related error, not a language error
            if response.status_code != 400 or "not supported" not in response.text:
                print_success(f"{lower_case} → accepted")
            else:
                print_error(f"{lower_case} → rejected")
                return False
        
        except Exception as e:
            print_error(f"Test failed: {str(e)}")
            return False
    
    return True

# ============================================
# MAIN TEST RUNNER
# ============================================

def run_all_tests(test_file: str = None) -> None:
    """Run all tests"""
    
    print_header("A-I Duo Voice Detection API - Test Suite")
    
    print_info(f"API URL: {API_URL}")
    print_info(f"API Key: {API_KEY}")
    print_info(f"Timeout: {TIMEOUT} seconds")
    
    test_results = {}
    
    # Basic tests
    print("\n" + Colors.BOLD + "Running basic tests..." + Colors.END)
    test_results["Health Check"] = test_health_check()
    test_results["Root Endpoint"] = test_root_endpoint()
    test_results["Invalid API Key"] = test_invalid_api_key()
    test_results["Invalid Language"] = test_invalid_language()
    test_results["Case-Insensitive Language"] = test_case_insensitive_language()
    
    # File upload tests (if file provided)
    if test_file and os.path.exists(test_file):
        print("\n" + Colors.BOLD + "Running file upload tests..." + Colors.END)
        test_results["File Upload (English)"] = test_file_upload(test_file, "English") is not None
        test_results["File Upload (Tamil)"] = test_file_upload(test_file, "Tamil") is not None
        test_results["Base64 Encoding"] = test_base64_encoding(test_file, "English") is not None
        test_results["Multi-language Test"] = test_multiple_languages(test_file)
    else:
        print_warning("No test audio file provided. Skipping file upload tests.")
        print_info("Usage: python test_api.py path/to/audio.mp3")
    
    # Summary
    print_header("Test Summary")
    
    passed = sum(1 for v in test_results.values() if v is True or isinstance(v, dict))
    total = len(test_results)
    
    for test_name, result in test_results.items():
        status = "PASSED" if result else "FAILED"
        symbol = "✓" if result else "✗"
        color = Colors.GREEN if result else Colors.RED
        print(f"{color}{symbol} {test_name}: {status}{Colors.END}")
    
    print(f"\n{Colors.BOLD}Total: {passed}/{total} tests passed{Colors.END}\n")
    
    if passed == total:
        print_success("All tests passed! API is working correctly.")
    else:
        print_warning(f"{total - passed} test(s) failed. Please review the results.")

# ============================================
# CLI INTERFACE
# ============================================

if __name__ == "__main__":
    
    print(f"\n{Colors.BOLD}{Colors.CYAN}A-I Duo Voice Detection API - Test Script{Colors.END}\n")
    
    test_file = sys.argv[1] if len(sys.argv) > 1 else None
    
    if test_file:
        print_info(f"Test audio file: {test_file}\n")
    
    try:
        run_all_tests(test_file)
    except KeyboardInterrupt:
        print_error("\n\nTests interrupted by user")
        sys.exit(1)
    except Exception as e:
        print_error(f"\n\nUnexpected error: {str(e)}")
        sys.exit(1)