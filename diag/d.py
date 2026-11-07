import subprocess, sys
sys.exit(subprocess.call([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "tests/test_kkt_report.py", "tests/test_linear_image.py", "tests/test_lbfgsb.py", "tests/test_radio_linear_image.py", "-x", "-q"]))
