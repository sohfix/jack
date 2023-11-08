import black
import pathlib


def format(directory_path: str):
    # Get all Python files in directory
    files = pathlib.Path(directory_path).rglob("*.py")

    try:
        # Format each file
        for file in files:
            with open(file, "r") as f:
                file_contents = f.read()
            formatted_contents = black.format_str(
                file_contents,
                mode=black.FileMode(),
            )
            with open(file, "w") as f:
                f.write(formatted_contents)
    except FileNotFoundError:
        FileNotFoundError("Do not working good today.")
