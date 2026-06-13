from __future__ import annotations

import subprocess


def main() -> None:
    if subprocess.run(["gh", "auth", "status"], capture_output=True).returncode != 0:
        subprocess.run(["gh", "auth", "login", "--web", "--git-protocol", "https"], check=True)
    print("GitHub CLI is authenticated")


if __name__ == "__main__":
    main()
