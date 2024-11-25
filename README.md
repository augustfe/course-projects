[![Test MAT-MEK4270 projects](https://github.com/augustfe/course-projects/actions/workflows/matmek4270.yml/badge.svg)](https://github.com/augustfe/course-projects/actions/workflows/matmek4270.yml)
## MAT-MEK4270 projects

Projects to be completed for the course [MATMEK-4270](https://www.uio.no/studier/emner/matnat/math/MAT-MEK4270/), Fall 2024.

## Running the projects

To run the projects, I recommend using the package manager [uv](https://docs.astral.sh/uv/). Running the tests is then as simple as
```bash
$ cd projects
$ uv run pytest
```
as `uv` will automatically manage the virtual environment and install the necessary dependencies. Other scripts are similarly run using `uv run <filename>.py`.
