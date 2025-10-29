## Contributing In General
Our project welcomes external contributions. If you have an itch, please feel
free to scratch it.

To contribute code or documentation, please submit a [pull request](https://github.com/docling-project/docling-jobkit/pulls).

A good way to familiarize yourself with the codebase and contribution process is
to look for and tackle low-hanging fruit in the [issue tracker](https://github.com/docling-project/docling-jobkit/issues).
Before embarking on a more ambitious contribution, please quickly [get in touch](#communication) with us.

For general questions or support requests, please refer to the [discussion section](https://github.com/docling-project/docling/discussions)
of the main Docling repository.

**Note: We appreciate your effort and want to avoid situations where a contribution
requires extensive rework (by you or by us), sits in the backlog for a long time, or
cannot be accepted at all!**

### Proposing New Features

If you would like to implement a new feature, please [raise an issue](https://github.com/docling-project/docling-jobkit/issues)
before sending a pull request so the feature can be discussed. This is to avoid
you spending valuable time working on a feature that the project developers
are not interested in accepting into the codebase.

### Fixing Bugs

If you would like to fix a bug, please [raise an issue](https://github.com/docling-project/docling/docling-jobkit) before sending a
pull request so it can be tracked.

### Merge Approval

The project maintainers use LGTM (Looks Good To Me) in comments on the code
review to indicate acceptance. A change requires LGTMs from two of the
maintainers of each component affected.

For a list of the maintainers, see the [MAINTAINERS.md](MAINTAINERS.md) page.


## Legal

Each source file must include a license header for the MIT
Software. Using the SPDX format is the simplest approach,
e.g.

```
/*
Copyright IBM Inc. All rights reserved.

SPDX-License-Identifier: MIT
*/
```

We have tried to make it as easy as possible to make contributions. This
applies to how we handle the legal aspects of contribution. We use the
same approach - the [Developer's Certificate of Origin 1.1 (DCO)](https://github.com/hyperledger/fabric/blob/master/docs/source/DCO1.1.txt) - that the Linux® Kernel [community](https://elinux.org/Developer_Certificate_Of_Origin)
uses to manage code contributions.

We simply ask that when submitting a patch for review, the developer
must include a sign-off statement in the commit message.

Here is an example Signed-off-by line, which indicates that the
submitter accepts the DCO:

```
Signed-off-by: John Doe <john.doe@example.com>
```

You can include this automatically when you commit a change to your
local git repository using the following command:

```
git commit -s
```

### New dependencies

This project strictly adheres to using dependencies that are compatible with the MIT license to ensure maximum flexibility and permissiveness in its usage and distribution. As a result, dependencies licensed under restrictive terms such as GPL, LGPL, AGPL, or similar are explicitly excluded. These licenses impose additional requirements and limitations that are incompatible with the MIT license's minimal restrictions, potentially affecting derivative works and redistribution. By maintaining this policy, the project ensures simplicity and freedom for both developers and users, avoiding conflicts with stricter copyleft provisions.


## Communication

Please feel free to connect with us using the [discussion section](https://github.com/docling-project/docling/discussions) of the main Docling repository.



## Developing

### Usage of `uv`

We use `uv` to manage dependencies.

#### Installation

To install `uv`, follow the documentation here: https://docs.astral.sh/uv/getting-started/installation/

1. Install `uv` globally on your machine:
    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```

3. The official guidelines linked above include useful details on configuring autocomplete for most shell environments, e.g., Bash and Zsh.

#### Create a Virtual Environment and Install Dependencies

To create the Virtual Environment, run:

```bash
uv venv
```

The virtual environment can be "activated" to make its packages available:

```bash
source .venv/bin/activate
```

Then, to install dependencies, run:

```bash
uv sync
```

**(Advanced) Use a Specific Python Version**

If you need to work with a specific (older) version of Python, run:

```bash
uv venv --python 3.11
```

More detailed options are described in the [uv documentation](https://docs.astral.sh/uv/pip/environments).


#### Add a New Dependency

```bash
uv add NAME
```

### Developing and testing the RQ engine

Here are the setup steps for the local development and testing of the RQ engine.

Launch the local redis container:

```sh
docker run -d --name redis -p 6379:6379 redis
```

Launch the workers (to be repeated when developing the worker code):

```sh
uv run python docling_jobkit/orchestrators/rq/worker.py
```


## Coding style guidelines

We use the following tools to enforce code style:

- ruff, to sort imports and format code

We run a series of checks on the code base on every commit, using `pre-commit`. To install the hooks, run:

```bash
pre-commit install
```

To run the checks on-demand, run:

```shell
pre-commit run --all-files
```

Note: Formatting checks like `ruff` will "fail" if they modify files. This is because `pre-commit` doesn't like to see files modified by their Hooks. In these cases, `git add` the modified files and `git commit` again.