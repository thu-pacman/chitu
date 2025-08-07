#!/usr/bin/env python3
"""
Multi-DP Configuration Availability Test Script

Comprehensive verification tool to check if multi-DP system is working properly
Supports DP=2, DP=4, DP=8 and other configurations

Usage:
python cinfer/test/validate_dp_req.py [--dp-size N] [--concurrent-tests N]
"""

import requests
import json
import time
import sys
import argparse
from concurrent.futures import ThreadPoolExecutor


def detect_dp_configuration(router_url):
    """Auto-detect DP configuration from router"""
    try:
        response = requests.get(f"{router_url}/dp/config", timeout=5)
        if response.status_code == 200:
            config = response.json()
            print(f"DP config: {config}")
            dp_size = config.get("scheduler_count", 2)
            connected_schedulers = config.get("scheduler_addresses", 0)
            return dp_size, connected_schedulers
        else:
            print(f"WARNING: Failed to detect DP config, assuming DP=2")
            return 2, 2
    except Exception as e:
        print(f"WARNING: DP config detection failed: {e}, assuming DP=2")
        return 2, 2


def test_dp_availability(dp_size=None, concurrent_tests=None):
    """Comprehensive test for multi-DP availability"""

    print("Multi-DP Configuration Availability Test")
    print("=" * 50)

    router_url = "http://localhost:21003"

    # Auto-detect DP configuration if not specified
    if dp_size is None:
        detected_dp_size, connected_schedulers = detect_dp_configuration(router_url)
        dp_size = detected_dp_size
        print(
            f"Auto-detected DP configuration: DP={dp_size}, Connected Schedulers={connected_schedulers}"
        )
    else:
        print(f"Testing with specified DP configuration: DP={dp_size}")

    # Set concurrent test count based on DP size if not specified
    if concurrent_tests is None:
        concurrent_tests = max(8, dp_size * 3)  # At least 8 tests, or 3x DP size

    print(f"Configuration: DP={dp_size}, Concurrent Tests={concurrent_tests}")
    print("=" * 50)

    # Test 1: Health check and system status
    print("1. Testing Router health status and system info...")
    try:
        response = requests.get(f"{router_url}/dp/test", timeout=10)
        if response.status_code == 200:
            data = response.json()
            print(f"SUCCESS: Router health check passed, resp:{data}")
            print(
                f"   - Connected schedulers: {data.get('connected_schedulers', 'unknown')}"
            )
            print(f"   - Router status: {data.get('router_status', 'unknown')}")
            print(
                f"   - Load balancer: {data.get('load_balancer_algorithm', 'unknown')}"
            )

            # Check if all expected schedulers are connected
            expected_schedulers = dp_size
            actual_schedulers = data.get("connected_schedulers", 0)
            if actual_schedulers < expected_schedulers:
                print(
                    f"   WARNING: Only {actual_schedulers}/{expected_schedulers} schedulers connected"
                )
        else:
            print(f"ERROR: Router health check failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
    except Exception as e:
        print(f"ERROR: Router connection failed: {e}")
        return False

    # Test 2: Basic inference functionality validation
    print("\n2. Testing basic inference functionality...")

    test_cases = [
        {
            "name": "Simple question test",
            "messages": [{"role": "user", "content": "请介绍你自己?"}],
            "max_tokens": 100,
            "temperature": 1,
        },
        {
            "name": "Math calculation test",
            "messages": [{"role": "user", "content": "15 + 27 = ?"}],
            "max_tokens": 100,
            "temperature": 1,
        },
        {
            "name": "Short story generation",
            "messages": [
                {"role": "user", "content": "写一个关于机器人的故事，2句话即可。"}
            ],
            "max_tokens": 100,
            "temperature": 1,
        },
        {
            "name": "Code completion test",
            "messages": [
                {
                    "role": "user",
                    "content": "Complete this Python function: def add_numbers(a, b):",
                }
            ],
            "max_tokens": 100,
            "temperature": 1,
        },
        {
            "name": "Question answering",
            "messages": [{"role": "user", "content": "宫保鸡丁怎么做?"}],
            "max_tokens": 100,
            "temperature": 1,
        },
    ]

    successful_tests = 0
    total_tokens_generated = 0

    for i, test_case in enumerate(test_cases, 1):
        print(f"\n   Test {i}: {test_case['name']}")
        print(f"      Input: {test_case['messages'][0]['content']}")

        try:
            test_request = {
                "messages": test_case["messages"],
                "max_tokens": test_case["max_tokens"],
                "temperature": test_case["temperature"],
                "stream": False,
            }

            start_time = time.time()
            response = requests.post(
                f"{router_url}/v1/chat/completions", json=test_request, timeout=30
            )
            end_time = time.time()

            if response.status_code == 200:
                data = response.json()
                response_time = end_time - start_time

                # Extract response content
                choices = data.get("choices", [])
                if choices:
                    message = choices[0].get("message", {})
                    content = message.get("content", "")

                    # Extract token statistics
                    usage = data.get("usage", {})
                    completion_tokens = usage.get("completion_tokens", 0)
                    prompt_tokens = usage.get("prompt_tokens", 0)
                    total_tokens = usage.get("total_tokens", 0)

                    if content and len(content.strip()) > 0:
                        print(f"      SUCCESS (Time: {response_time:.2f}s)")
                        print(
                            f"      Output: {content[:100]}{'...' if len(content) > 100 else ''}"
                        )
                        print(
                            f"      Tokens: input={prompt_tokens}, generated={completion_tokens}, total={total_tokens}"
                        )
                        successful_tests += 1
                        total_tokens_generated += completion_tokens
                    else:
                        print(f"      WARNING: Empty response received")
                        print(
                            f"      Full response: {json.dumps(data, ensure_ascii=False, indent=2)}"
                        )
                else:
                    print(f"      ERROR: Invalid response format - no choices field")
                    print(
                        f"      Full response: {json.dumps(data, ensure_ascii=False, indent=2)}"
                    )
            else:
                print(f"      ERROR: Request failed with HTTP {response.status_code}")
                print(f"      Error response: {response.text}")

        except Exception as e:
            print(f"      ERROR: Request exception: {e}")

    # Test 3: High-concurrency stress test
    print(
        f"\n3. Testing high-concurrency processing (DP={dp_size}, {concurrent_tests} concurrent requests)..."
    )

    def send_concurrent_request(req_id):
        try:
            # Vary request types to test different processing patterns
            request_types = [
                {"content": f"Request #{req_id}: 15 + 27 = ?", "max_tokens": 100},
                {
                    "content": f"Request #{req_id}: 写一个关于机器人的故事，2句话即可。",
                    "max_tokens": 100,
                },
                {"content": f"Request #{req_id}: 宫保鸡丁怎么做?", "max_tokens": 100},
                {"content": f"Request #{req_id}: 请介绍你自己?", "max_tokens": 100},
                {
                    "content": f"Request #{req_id}: Count from 1 to {min(req_id % 5 + 1, 3)}.",
                    "max_tokens": 100,
                },
            ]

            request_type = request_types[req_id % len(request_types)]

            test_request = {
                "messages": [{"role": "user", "content": request_type["content"]}],
                "max_tokens": request_type["max_tokens"],
                "temperature": 1,
                "stream": False,
            }

            start_time = time.time()
            response = requests.post(
                f"{router_url}/v1/chat/completions", json=test_request, timeout=20
            )
            end_time = time.time()

            if response.status_code == 200:
                data = response.json()
                print(f"      Response: {data}")
                content = (
                    data.get("choices", [{}])[0].get("message", {}).get("content", "")
                )
                usage = data.get("usage", {})

                if content and len(content.strip()) > 0:
                    return {
                        "success": True,
                        "response_time": end_time - start_time,
                        "content": content,
                        "prompt_tokens": usage.get("prompt_tokens", 0),
                        "completion_tokens": usage.get("completion_tokens", 0),
                        "request_type": request_type["content"][:30],
                        "req_id": req_id,
                    }
            return {
                "success": False,
                "req_id": req_id,
                "status_code": response.status_code,
            }
        except Exception as e:
            return {"success": False, "req_id": req_id, "error": str(e)}

    # Execute concurrent requests
    print(f"   Launching {concurrent_tests} concurrent requests...")
    start_concurrent = time.time()

    with ThreadPoolExecutor(max_workers=min(concurrent_tests, 20)) as executor:
        futures = [
            executor.submit(send_concurrent_request, i) for i in range(concurrent_tests)
        ]
        results = [future.result() for future in futures]

    end_concurrent = time.time()
    total_concurrent_time = end_concurrent - start_concurrent

    # Analyze concurrent test results
    successful_results = [r for r in results if r.get("success", False)]
    failed_results = [r for r in results if not r.get("success", False)]

    concurrent_success = len(successful_results)

    if concurrent_success > 0:
        successful_times = [r["response_time"] for r in successful_results]
        avg_time = sum(successful_times) / len(successful_times)
        min_time = min(successful_times)
        max_time = max(successful_times)
        total_concurrent_tokens = sum(
            r.get("completion_tokens", 0) for r in successful_results
        )

        print(f"   SUCCESS: Concurrent test completed")
        print(
            f"   - Successful requests: {concurrent_success}/{concurrent_tests} ({concurrent_success/concurrent_tests*100:.1f}%)"
        )
        print(f"   - Total wall-clock time: {total_concurrent_time:.2f}s")
        print(
            f"   - Average response time: {avg_time:.2f}s (min: {min_time:.2f}s, max: {max_time:.2f}s)"
        )
        print(f"   - Total generated tokens: {total_concurrent_tokens}")
        print(
            f"   - Effective throughput: {total_concurrent_tokens/total_concurrent_time:.1f} tokens/second"
        )

        # Show sample successful responses
        print(f"   - Sample responses:")
        for i, result in enumerate(successful_results[:3]):
            print(f"     #{result['req_id']}: {result.get('content', '')[:60]}...")

        # Analyze failed requests if any
        if failed_results:
            print(f"   - Failed request analysis:")
            error_counts = {}
            for result in failed_results:
                error_type = result.get(
                    "error", f"HTTP_{result.get('status_code', 'unknown')}"
                )
                error_counts[error_type] = error_counts.get(error_type, 0) + 1

            for error_type, count in error_counts.items():
                print(f"     {error_type}: {count} requests")
    else:
        print(f"   ERROR: All concurrent requests failed")
        print(f"   - Failed requests: {len(failed_results)}/{concurrent_tests}")

        # Show failure details
        for result in failed_results[:5]:  # Show first 5 failures
            error_info = result.get(
                "error", f"HTTP {result.get('status_code', 'unknown')}"
            )
            print(f"   - Request #{result['req_id']}: {error_info}")

    # Test 4: Load balancing verification
    print(f"\n4. Testing load balancing across {dp_size} schedulers...")

    # Send requests to verify distribution
    balancing_tests = min(dp_size * 4, 20)  # 4 requests per scheduler, max 20
    print(f"   Sending {balancing_tests} requests to verify scheduler distribution...")

    balancing_results = []
    for i in range(balancing_tests):
        try:
            test_request = {
                "messages": [
                    {
                        "role": "user",
                        "content": f"Balance test {i+1}: reply with 'OK {i+1}'",
                    }
                ],
                "max_tokens": 5,
                "temperature": 0.1,
                "stream": False,
            }

            response = requests.post(
                f"{router_url}/v1/chat/completions", json=test_request, timeout=15
            )

            if response.status_code == 200:
                balancing_results.append({"success": True, "request_id": i + 1})
            else:
                balancing_results.append({"success": False, "request_id": i + 1})
        except Exception as e:
            balancing_results.append(
                {"success": False, "request_id": i + 1, "error": str(e)}
            )

    balancing_success = sum(1 for r in balancing_results if r.get("success", False))
    print(
        f"   Load balancing test: {balancing_success}/{balancing_tests} requests successful"
    )

    if balancing_success >= balancing_tests * 0.8:  # 80% success rate
        print(f"   SUCCESS: Load balancing appears to be working")
    else:
        print(f"   WARNING: Load balancing may have issues")

    # Final Summary
    print(f"\nTest Summary:")
    print("=" * 50)
    print(f"System Configuration: DP={dp_size}")
    print(f"Basic inference tests: {successful_tests}/{len(test_cases)} passed")
    print(
        f"Concurrent processing: {concurrent_success}/{concurrent_tests} successful ({concurrent_success/concurrent_tests*100:.1f}%)"
    )
    print(f"Load balancing test: {balancing_success}/{balancing_tests} successful")
    print(
        f"Total tokens generated: {total_tokens_generated + sum(r.get('completion_tokens', 0) for r in successful_results)}"
    )

    # Determine overall success criteria
    basic_threshold = len(test_cases) * 0.8  # 80% of basic tests
    concurrent_threshold = concurrent_tests * 0.7  # 70% of concurrent tests
    balancing_threshold = balancing_tests * 0.6  # 60% of balancing tests

    overall_success = (
        successful_tests >= basic_threshold
        and concurrent_success >= concurrent_threshold
        and balancing_success >= balancing_threshold
    )

    if overall_success:
        print(f"\nSUCCESS: DP={dp_size} system is working normally!")
        print(f"   - Basic functionality: PASS")
        print(f"   - Concurrent processing: PASS")
        print(f"   - Load balancing: PASS")
        return True
    else:
        print(f"\nERROR: DP={dp_size} system has issues:")
        if successful_tests < basic_threshold:
            print(
                f"   - Basic functionality: FAIL ({successful_tests}/{len(test_cases)})"
            )
        if concurrent_success < concurrent_threshold:
            print(
                f"   - Concurrent processing: FAIL ({concurrent_success}/{concurrent_tests})"
            )
        if balancing_success < balancing_threshold:
            print(f"   - Load balancing: FAIL ({balancing_success}/{balancing_tests})")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Multi-DP Configuration Availability Test"
    )
    parser.add_argument(
        "--dp-size",
        type=int,
        default=None,
        help="Expected DP size (auto-detected if not specified)",
    )
    parser.add_argument(
        "--concurrent-tests",
        type=int,
        default=None,
        help="Number of concurrent tests (auto-calculated if not specified)",
    )
    parser.add_argument(
        "--router-url",
        type=str,
        default="http://localhost:21003",
        help="Router URL (default: http://localhost:21003)",
    )

    args = parser.parse_args()

    try:
        success = test_dp_availability(args.dp_size, args.concurrent_tests)
        if success:
            print("\nAll tests passed - system working normally")
            sys.exit(0)
        else:
            print("\nTests failed - please check system status")
            sys.exit(1)
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nUnexpected error during testing: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
