"""Offline checks with bounded tool output and durable exit evidence."""
import argparse
import json
import os
from pathlib import Path
import sys
import traceback
import unittest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--pattern', default='test_*.py')
    args = parser.parse_args()
    run_dir = Path(args.run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / 'worker-pid.txt').write_text(str(os.getpid()))
    project = Path(__file__).resolve().parent.parent
    os.chdir(project)
    sys.path.insert(0, str(project))
    code = 1
    with (run_dir / 'checks.log').open('w', encoding='utf-8') as log:
        sys.stdout = sys.stderr = log
        try:
            suite = unittest.defaultTestLoader.discover(str(project / 'tests'), pattern=args.pattern)
            result = unittest.TextTestRunner(stream=log, verbosity=2).run(suite)
            code = 0 if result.wasSuccessful() else 1
            (run_dir / 'summary.json').write_text(json.dumps({
                'tests': result.testsRun, 'failures': len(result.failures),
                'errors': len(result.errors), 'skipped': len(result.skipped),
                'exit_code': code,
            }), encoding='utf-8')
        except BaseException:
            traceback.print_exc(file=log)
        finally:
            log.flush()
            (run_dir / 'exit-code.txt').write_text(str(code))
    return code


if __name__ == '__main__':
    raise SystemExit(main())
