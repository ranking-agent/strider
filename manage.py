#!/usr/bin/env python3
import sys
import os


def print_green(s):
    GREEN = '\033[92m'
    ENDC = '\033[0m'
    print(f"{GREEN}{s}{ENDC}")


def run_command(cmd):
    print_green(cmd)
    os.system(cmd)


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
    mkdir /tmp/strider-benchmark
    docker cp strider-testing:/app/report.png /tmp/strider-benchmark/report.png
    open /tmp/strider-benchmark/report.png
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


def lock(extra_args):
    """
    Write requirements-lock.txt and requirements-test-lock.txt
    """
    requirements_files = {
        "requirements.txt": "requirements-lock.txt",
        "requirements-test.txt": "requirements-test-lock.txt",
    }

    for src, locked in requirements_files.items():
        command = f"""\
        docker run -v $(pwd):/app python:3.9 \
            /bin/bash -c "pip install -qqq -r /app/{src} && pip freeze" > {locked}
        """
        run_command(command)


def main():
    command = sys.argv[1]
    command_func = globals()[command]
    extra_args = " " + " ".join(sys.argv[2:])
    command_func(extra_args)


if __name__ == '__main__':
    main()
