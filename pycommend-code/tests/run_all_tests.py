"""
Script to run all tests and generate baseline report
"""

import sys
import os
import subprocess
import datetime
import json
from pathlib import Path

def run_test_module(module_name):
    """Run a test module and capture results"""
    print(f"\n{'='*60}")
    print(f"Running {module_name}...")
    print('='*60)

    try:
        result = subprocess.run(
            [sys.executable, f'tests/{module_name}'],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent,
            timeout=60
        )

        # Parse output to get test results
        output = result.stdout + result.stderr

        # Look for test summary
        tests_run = 0
        failures = 0
        errors = 0
        skipped = 0

        for line in output.split('\n'):
            if 'Tests run:' in line:
                tests_run = int(line.split(':')[1].strip())
            elif 'Failures:' in line and 'TEST SUMMARY' in output:
                failures = int(line.split(':')[1].strip())
            elif 'Errors:' in line and 'TEST SUMMARY' in output:
                errors = int(line.split(':')[1].strip())
            elif 'Skipped:' in line and 'TEST SUMMARY' in output:
                skipped = int(line.split(':')[1].strip())
            elif 'Ran' in line and 'test' in line:
                # Fallback parser for unittest output
                parts = line.split()
                if 'Ran' in parts:
                    idx = parts.index('Ran')
                    if idx + 1 < len(parts):
                        tests_run = int(parts[idx + 1])

        # Check for FAILED in output
        if 'FAILED' in output:
            # Parse FAILED (failures=X, errors=Y)
            for line in output.split('\n'):
                if 'FAILED' in line and '(' in line:
                    failure_info = line[line.index('(')+1:line.index(')')]
                    for part in failure_info.split(','):
                        part = part.strip()
                        if 'failures=' in part:
                            failures = int(part.replace('failures=', ''))
                        elif 'errors=' in part:
                            errors = int(part.replace('errors=', ''))

        success = (failures == 0 and errors == 0) if tests_run > 0 else False

        return {
            'module': module_name,
            'tests_run': tests_run,
            'failures': failures,
            'errors': errors,
            'skipped': skipped,
            'success': success,
            'return_code': result.returncode
        }

    except subprocess.TimeoutExpired:
        print(f"Test {module_name} timed out!")
        return {
            'module': module_name,
            'tests_run': 0,
            'failures': 0,
            'errors': 1,
            'skipped': 0,
            'success': False,
            'return_code': -1
        }
    except Exception as e:
        print(f"Error running {module_name}: {e}")
        return {
            'module': module_name,
            'tests_run': 0,
            'failures': 0,
            'errors': 1,
            'skipped': 0,
            'success': False,
            'return_code': -1
        }


def main():
    """Run all tests and generate baseline report"""
    print("PyCommend Test Suite - Baseline Report")
    print(f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Change to project directory
    os.chdir(Path(__file__).parent.parent)

    # Test modules to run
    test_modules = [
        'test_preprocessor.py',
        'test_nsga2.py'
    ]

    results = []
    for module in test_modules:
        result = run_test_module(module)
        results.append(result)

    # Generate summary
    print("\n" + "="*60)
    print("BASELINE TEST SUMMARY")
    print("="*60)

    total_tests = sum(r['tests_run'] for r in results)
    total_failures = sum(r['failures'] for r in results)
    total_errors = sum(r['errors'] for r in results)
    total_skipped = sum(r['skipped'] for r in results)

    print(f"Total tests run: {total_tests}")
    print(f"Total failures: {total_failures}")
    print(f"Total errors: {total_errors}")
    print(f"Total skipped: {total_skipped}")
    print()

    # Module breakdown
    print("Module Breakdown:")
    for result in results:
        status = "PASS" if result['success'] else "FAIL"
        print(f"  - {result['module']}: {status}")
        print(f"    Tests: {result['tests_run']}, Failures: {result['failures']}, "
              f"Errors: {result['errors']}, Skipped: {result['skipped']}")

    # Overall status
    all_pass = all(r['success'] for r in results)
    overall_status = "PASS" if all_pass else "FAIL"

    print()
    print(f"Overall Status: {overall_status}")

    # Save baseline report
    baseline_report = {
        'date': datetime.datetime.now().isoformat(),
        'summary': {
            'total_tests': total_tests,
            'total_failures': total_failures,
            'total_errors': total_errors,
            'total_skipped': total_skipped,
            'overall_status': overall_status
        },
        'modules': results
    }

    report_file = Path('tests/baseline_report.json')
    with open(report_file, 'w') as f:
        json.dump(baseline_report, f, indent=2)

    print(f"\nBaseline report saved to: {report_file}")

    # Return appropriate exit code
    sys.exit(0 if all_pass else 1)


if __name__ == '__main__':
    main()