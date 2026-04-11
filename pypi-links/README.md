Some packages only exist in official PyPI index, instead of its mirrors. In order to get these packages from official PyPI index, while getting other packages from mirrors, we copy necessary part of index from official PyPI site here.

To use, set `--find-links <this-directory>` to `pip`.

Please note that we can't simply set `--extra-index-url https://pypi.org/simple` to `pip` in addition to mirrors, because `pip` does not search index-urls in order.
