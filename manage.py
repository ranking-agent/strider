#!/usr/bin/env python3
import sys
import os


def print_green(s):
    GREEN = "\033[92m"
    ENDC = "\033[0m"
    print(f"{GREEN}{s}{ENDC}")


def run_command(cmd):
    print_green(cmd)
    os.system(cmd)


def get_command_output(cmd):
    return os.popen(cmd).read()


def dev(extra_args):
    """
    This command starts up a development environment.
    The development environment is started through docker-compose
    and is visible at http://localhost
    """
    command = f"docker-compose -f docker-compose.yml -f docker-compose.dev.yml \
                up --build --renew-anon-volumes"
    run_command(command + extra_args)


def test(extra_args):
    """
    This command runs the tests within docker
    and then exits.
    """
    command = """\
    docker build -t strider-testing -f Dockerfile.test .
    docker run -it strider-testing\
    """
    run_command(command + extra_args)


def benchmark(extra_args):
    """
    This command runs benchmarks within docker
    and displays them
    """
    command = """\
    docker rm strider-testing || true
    docker build -t strider-testing \
                 -f Dockerfile.test .
    docker run --name strider-testing strider-testing \
            python -m benchmarks {extra_args}
    """
    run_command(command + extra_args)


def coverage(extra_args):
    """
    Run tests in docker, copy out a coverage report,
    and display in browser
    """
    command = f"""\
    docker rm strider-testing || true
    docker build -t strider-testing \
                 -f Dockerfile.test .
    docker run --name strider-testing strider-testing \
            pytest --cov strider/ --cov-report html {extra_args}
    docker cp strider-testing:/app/htmlcov /tmp/strider-cov/
    open /tmp/strider-cov/index.html
    """
    run_command(command)


def profile(extra_args):
    """
    Profile a test in docker, copy out a report,
    and display using the snakeviz utility
    """
    command = f"""\
    docker rm strider-profile || true
    docker build -t strider-profile \
                 -f Dockerfile.test .
    docker run --name strider-profile strider-profile \
            python -m cProfile -o strider.prof -m pytest {extra_args}
    docker cp strider-profile:/app/strider.prof /tmp/
    snakeviz /tmp/strider.prof
    """
    run_command(command)


REQUIREMENTS_FILES = {
    "requirements.txt": "requirements-lock.txt",
    "requirements-test.txt": "requirements-test-lock.txt",
}


def lock(extra_args):
    """
    Write lockfiles without upgrading dependencies
    """
    for src, locked in REQUIREMENTS_FILES.items():
        command = f"""\
        docker run -v $(pwd):/app python:3.9 \
            /bin/bash -c "
                # Install lockfile first so that we get the
                # currently installed versions of dependencies
                pip install -r /app/{locked} &&
                pip install -r /app/{src}    &&
                # Write lockfile
                pip freeze > /app/{locked}
            "
        """
        run_command(command)


def verify_locked(extra_args):
    """
    Verify that the lockfile is up-to-date
    """

    for src, locked in REQUIREMENTS_FILES.items():
        dependencies = get_command_output(
            f"""\
        docker run -v $(pwd):/app python:3.9 \
            /bin/bash -c "
                pip install -qqq -r /app/{locked} &&
                pip install -qqq -r /app/{src}    &&
                pip freeze
            "
        """
        )
        lock_dependencies = get_command_output(
            f"""\
        docker run -v $(pwd):/app python:3.9 \
            /bin/bash -c "
                pip install -qqq -r /app/{locked} &&
                pip freeze
            "
        """
        )

        if dependencies != lock_dependencies:
            sys.exit(f"Lock file {locked} not up-to-date, please run ./manage.py lock")


def upgrade(extra_args):
    """
    Upgrade all dependencies
    """
    for src, locked in REQUIREMENTS_FILES.items():
        command = f"""\
        docker run -v $(pwd):/app python:3.9 \
            /bin/bash -c "
                # Install dependencies, getting latest version
                pip install -r /app/{src} &&
                # Write lockfile
                pip freeze > /app/{locked}
            "
        """
        run_command(command)


def main():
    command = sys.argv[1]
    command_func = globals()[command]
    extra_args = " " + " ".join(sys.argv[2:])
    command_func(extra_args)


if __name__ == "__main__":
    main()
