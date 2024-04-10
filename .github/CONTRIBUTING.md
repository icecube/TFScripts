# Contributing

Welcome and thanks for considering to contribute to this repository!

## Pre-commit Hooks

When contributing to this project, please utilize the pre-commit hooks. If not installed yet, you will need to add the python package `pre-commit`:

    pip install pre-commit

Once the package is installed, simply install the pre-commit hooks defined in the repository by executing:

    pre-commit install

from within the repository directory.

The pre-commit hooks will now automatically run when invoking `git commit`. Note, however, that this requires an active shell that has `pre-commit` installed.
You can also manually run the pre-commit on single files or on all files via:

    pre-commit run --all-files

If you need to commit something even though there are errors (this should not have to be done!), then you can add the flag `--no-verify` to the `git commit` command. This will bypass the pre-commit hooks.

Additional information is provided here: https://pre-commit.com/
