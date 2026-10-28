This is the Chitu repo. Chitu is an LLM inference engine. See `README.md` for details.

# Before committing

1. Check "Coding style rules" below.
2. Run `mypy chitu` for static type checking. It is only a hint for you, and you don't need to fix them all. You don't need to suppress them or work around them either. Only fix real bugs in the code you have modified.
3. Use `black` to format Python code. Use `clang-format` to format C-like (C/C++/CUDA) code.
4. Run `./script/generate_cli_docs.py`, `./script/generate_http_api_docs.py`, `./script/generate_metrics_docs.py`, and `./script/generate_supported_models_docs.py` to update auto-generated docs.

# Coding style rules

When developing this repo, follow the rules below. Before commiting, check your diff against the rules again.

## Comments are for future developers

Comments should describe the fact, reasons, and pitfalls of the code. Comments shout NOT be used as a diff log. Readers do NOT care about the difference before and after your change.

Good comment:

```
# Set `c=3` because case X needs Y.
f(a, b, c=3)
```

Bad comment:

```
# Keep `a` and `b` as historical code. Fix it by adding `c=3`.
f(a, b, c=3)
```

## Organize sentences in comments by topics

One topic per sentences, rather than one object and all its related topic in one sentence.

Good comment:

```
# Reasoning is separated by delimiters. A delimiter may consist of multiple tokens. E.g., an
# XTML-like delimiter for model X.
#
# Reasoning is switchable per request: When on, the output starts from inside thinking channel.
# When off, the output starts from outside thinking channel.
```

Bad comment:

```
# Reasoning - switchable per request (thinking on/off), maybe with a multi-token channel
# delimiter (XTML like for model X) - makes output starting from inside thinking channel
# (thinking=True), or before thinking channel (thinking=False).
```

## Use unambiguous and consistent naming

Use the same name for the same thing. Use different names for different things.

## User-facing configs should be worth tuning by users

When you want to add a new item in a config file, or a new environment variable, or a new command-line option, please consider the following.

You can add a user-facing config in the following cases:
1. You know sometimes one option is better and sometime another option is better, but you don't know eactly when, so you want the users to decide. Example: what kind of parallelism to use.
2. There is no better or worse on this config, but it depends on user preferences. Example: verbose level of logging.

Do NOT add a user-facing config in the following case:
1. You know exactly when to set the config to what value, or you know it is always better to set to a specific value. In this case, you should decide it in a programmatic way instead.
2. The config is only for a temporary probe or experiment, but not for production. In this case, drop the config.
3. The config is only for a permanent test, but not for production. In this case, you should implement a mock in the test instead.

After checking the rules above, if you still think the config is needed, please consider its form:
1. A new item in an existing Hydra config file is preferred, because we have established tooling around Hydra config files.
2. Otherwise, if you are editing a standalone test or installation script, a command line argument is preferred.
3. Environment variables are last resort, for special cases like options handled before Hydra initialization. If you really need to add an environment variable, please also update the environment variable sections in related documents.

## Fold redundant code

Bad example (redundant variable passing):

```
sum = a + b
result = sum
return result
```

Good example (no redundant variable passing):

```
return a + b
```

Bad example (redundant function call):

```
def _sum(a, b):
    return a + b

def sum(a, b):
    return _sum(a, b)
```

Good example (no redundant function call):

```
def sum(a, b):
    return a + b
```

## Defensive programming is not to hide bugs, but to expose them

Good example (raise on unsupported cases):

```
def sqrt(x):
    if x >= 0:
        ... # compute arithmetic square root
    else:
        raise ValueError(f"x ({x}) cannot be a negative value")
```

Bad example (output incorrect result in unsupported cases):

```
def sqrt(x):
    x = abs(x)
    ... # compute arithmetic square root
```
