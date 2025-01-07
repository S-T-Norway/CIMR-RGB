# System Tests

> This folder contains system/varification tests.

To run the test, one should do (from the root of the repo!):

```
$ pytest tests/system
```

If one wants to run a specific test, one can do:

```
$ pytest tests/system/test_T.12.py
```

or if one wants to run a bunch of tests, one can do so like this:

```
$ pytest tests/system/test_T.{12,13,14}.py

```

Now, the above lines will NOT show the output of the entire run to the
terminal. To get it, one should add `-s` parameter at the end for this to happen, like so:

```
$ pytest tests/system/test_T.{12,13,14}.py -s
```
