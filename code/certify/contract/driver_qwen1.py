"""One Qwen engine load -> run E1 (build store_e1 + certify) then E5 (currency).
Foreground use only (offline in-process backend). Reuses the module engine singleton."""
import sys, os
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import certify, currency

print("=== E1 (Qwen build + certify) ===", flush=True)
sys.argv = ["certify.py", "--workers", "2"]
certify.main()

print("\n=== E5 (currency, Qwen) ===", flush=True)
sys.argv = ["currency.py", "--workers", "2", "--batches", "8"]
currency.main()
print("\n=== driver_qwen1 done ===", flush=True)
