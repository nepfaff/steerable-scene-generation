import os
import re


def extract_output_dir_from_stdout(stdout: str) -> str | None:
    """Extract the `output_dir` from the script's stdout using regex."""
    # First, strip ANSI escape sequences.
    clean_stdout = re.sub(r"\x1b\[[0-9;]*[a-zA-Z]", "", stdout)
    match = re.search(r"Outputs will be saved to:\s*(.+)", clean_stdout)
    if match:
        return match.group(1).strip()
    return None


def find_first_ckpt_file(checkpoint_dir: str) -> str | None:
    """Extract the first .ckpt file in checkpoint_dir."""
    # List all files in the directory.
    for file_name in os.listdir(checkpoint_dir):
        # Check if the file ends with .ckpt.
        if file_name.endswith(".ckpt"):
            # Return the full path of the first .ckpt file.
            return os.path.join(checkpoint_dir, file_name)
    return None
