Distributed PyTest test cases.

Usage:

```
torchrun [torchrun arguments...] --no-python ./run_pytest_with_pretty_print.sh [pytest arguments...] ./
```

Since errors in distributed programs often make communications hang, it is recommended to make `pytest` exit on the first error by setting `-x` as a pytest argument.
