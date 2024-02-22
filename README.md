# Attention Controlled Electromagnetic Levitation System


Welcome to the A.C.E.L.S package!

This package holds the tool that was developed to control an electromagnetic levitation system.


#### Installation of Poetry
The installation instructions for poetry can be found [on their website](https://python-poetry.org/docs/).  Don't forget to add the new installation directory into your PATH.

#### Installing the package using poetry

To install all dependencies for `acels` simply run `poetry install`. Some basic usage and commands for poetry can be found [here](https://python-poetry.org/docs/cli/) or by running `poetry help`.

With poetry, commands are executed inside the virtual environment and to signify this each command should be pre-pended with `poetry run`. For example to run pytest on the `acels` package you should run:
```sh
poetry run pytest
```
After `poetry run` you can specify any commands and arguments you need.

You can also run `poetry shell` which opens a poetry environment within the terminal. You can also integrate this into the Python interpreters in VS Code. Start by finding the path of the Python interpreter you're using (for instance, by running `which python` on Linux). Then, input this path into the "Select Interpreter" section of VS Code settings.