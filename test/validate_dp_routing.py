#!/usr/bin/env python3
"""
DP Routing Validation Script

Validates whether requests are correctly routed to different Schedulers
"""

import requests
import time
import threading
from collections import Counter


def test_routing_distribution():
    """Test request routing distribution"""

    print("DP Routing Distribution Validation Test")
    print("=" * 50)

    router_url = "http://localhost:21003"

    # Test 1: Check system status
    print("1. Check system status...")
    try:
        response = requests.get(f"{router_url}/dp/debug", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print("PASS: System status check")
            print(f"   - DP enabled: {data.get('dp_enabled', 'unknown')}")
            print(
                f"   - DP service status: {data.get('dp_service_started', 'unknown')}"
            )

            # Check Router status
            if "request_router" in data:
                router_info = data["request_router"]
                print(
                    f"   - Pending requests: {router_info.get('pending_requests', 'unknown')}"
                )
                print(
                    f"   - Scheduler stats: {router_info.get('scheduler_stats', 'unknown')}"
                )
        else:
            print(f"FAIL: System status check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"FAIL: Cannot connect to Router: {e}")
        return False

    # Test 2: Send multiple sequential requests to observe routing distribution
    print("\n2. Sequential request routing distribution test...")

    def send_single_request(req_id):
        try:
            test_request = {
                "messages": [
                    {
                        "role": "user",
                        "content": f"Request #{req_id}: What is 1+1? Please answer briefly.",
                    }
                ],
                "max_tokens": 10,
                "temperature": 0.1,
                "stream": False,
            }

            start_time = time.time()
            response = requests.post(
                f"{router_url}/v1/chat/completions", json=test_request, timeout=20
            )
            end_time = time.time()

            if response.status_code == 200:
                return {
                    "success": True,
                    "req_id": req_id,
                    "response_time": end_time - start_time,
                    "status": "success",
                }
            else:
                return {
                    "success": False,
                    "req_id": req_id,
                    "status": f"HTTP {response.status_code}",
                    "error": response.text[:100],
                }
        except Exception as e:
            return {
                "success": False,
                "req_id": req_id,
                "status": "exception",
                "error": str(e),
            }

    # Send 10 sequential requests
    print("   Sending 10 sequential requests...")
    results = []
    for i in range(10):
        print(f"   Sending request #{i+1}...")
        result = send_single_request(i + 1)
        results.append(result)
        time.sleep(0.5)  # 500ms interval

    # Statistics
    successful_requests = [r for r in results if r["success"]]
    failed_requests = [r for r in results if not r["success"]]

    print(f"\n   Sequential request results:")
    print(f"   - Success: {len(successful_requests)}/10")
    print(f"   - Failed: {len(failed_requests)}/10")

    if successful_requests:
        avg_time = sum(r["response_time"] for r in successful_requests) / len(
            successful_requests
        )
        print(f"   - Average response time: {avg_time:.2f}s")

    if failed_requests:
        print(f"\n   Failed request details:")
        for fail in failed_requests[:3]:  # Show only first 3 failures
            print(
                f"      Request #{fail['req_id']}: {fail['status']} - {fail.get('error', '')[:50]}"
            )

    # Test 3: Concurrent request test to observe load balancing
    print("\n3. Concurrent request load balancing test...")

    def send_concurrent_batch(batch_id, batch_size=3):
        batch_results = []
        threads = []

        def worker(req_id):
            result = send_single_request(f"{batch_id}-{req_id}")
            batch_results.append(result)

        # Start concurrent threads
        for i in range(batch_size):
            t = threading.Thread(target=worker, args=(i + 1,))
            threads.append(t)
            t.start()

        # Wait for all threads to complete
        for t in threads:
            t.join()

        return batch_results

    print("   Sending 3 batches of concurrent requests, 3 requests per batch...")
    all_concurrent_results = []

    for batch in range(3):
        print(f"   Batch #{batch+1}...")
        batch_results = send_concurrent_batch(batch + 1)
        all_concurrent_results.extend(batch_results)
        time.sleep(1)  # 1 second interval between batches

    # Concurrent results statistics
    concurrent_successful = [r for r in all_concurrent_results if r["success"]]
    concurrent_failed = [r for r in all_concurrent_results if not r["success"]]

    print(f"\n   Concurrent request results:")
    print(f"   - Success: {len(concurrent_successful)}/9")
    print(f"   - Failed: {len(concurrent_failed)}/9")

    if concurrent_successful:
        concurrent_avg_time = sum(
            r["response_time"] for r in concurrent_successful
        ) / len(concurrent_successful)
        print(f"   - Average response time: {concurrent_avg_time:.2f}s")

    # Test 4: Check final system status
    print("\n4. Check post-test system status...")
    try:
        response = requests.get(f"{router_url}/dp/debug", timeout=5)
        if response.status_code == 200:
            data = response.json()
            if "request_router" in data:
                router_info = data["request_router"]
                total_requests = router_info.get("total_requests", 0)
                print(f"   PASS: Total processed requests: {total_requests}")

                scheduler_stats = router_info.get("scheduler_stats", {})
                if scheduler_stats:
                    print(f"   Scheduler statistics:")
                    for scheduler_id, stats in scheduler_stats.items():
                        print(
                            f"      Scheduler {scheduler_id}: running={stats.get('running_requests', 0)}, "
                            f"waiting={stats.get('waiting_requests', 0)}, "
                            f"throughput={stats.get('throughput_tokens_per_sec', 0):.2f}"
                        )
                else:
                    print(f"   WARNING: No scheduler statistics available")
    except Exception as e:
        print(f"   WARNING: Cannot get final status: {e}")

    # Summary
    print(f"\nTest Summary:")
    print("=" * 50)
    total_successful = len(successful_requests) + len(concurrent_successful)
    total_requests = len(results) + len(all_concurrent_results)
    success_rate = (
        (total_successful / total_requests) * 100 if total_requests > 0 else 0
    )

    print(f"Total requests: {total_requests}")
    print(f"Successful requests: {total_successful}")
    print(f"Success rate: {success_rate:.1f}%")

    if success_rate >= 80:
        print("\nPASS: DP routing system is working normally!")
        if success_rate >= 95:
            print("   EXCELLENT: System performance is excellent")
        return True
    else:
        print(f"\nFAIL: DP routing system may have issues")
        print(f"   Suggestion: Check logs for detailed error information")
        return False


def main():
    try:
        print("Starting DP routing validation test...\n")
        success = test_routing_distribution()

        if success:
            print("\nPASS: Validation test completed, system working normally")
            print("\nTip: Check system logs to observe specific routing distribution")
            print(
                "     Logs should show requests being assigned to different Schedulers"
            )
            exit(0)
        else:
            print("\nFAIL: Validation test found issues")
            exit(1)

    except KeyboardInterrupt:
        print("\nWARNING: Test interrupted by user")
        exit(1)
    except Exception as e:
        print(f"\nFAIL: Exception occurred during test: {e}")
        exit(1)


if __name__ == "__main__":
    main()
