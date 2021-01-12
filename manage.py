import click
import os


def print_green(s):
    GREEN = '\033[92m'
    ENDC = '\033[0m'
    print(f"{GREEN}{s}{ENDC}")


@click.group()
def main():
    pass


@main.command()
def dev():
    """ This command starts up a development environment.

    The development environment is started through docker-compose and is visible
    at http://localhost:.
    """
    command = f"docker-compose -f docker-compose.yml -f docker-compose.dev.yml \
                up --build --renew-anon-volumes"
    print_green(command)
    os.system(command)


@main.command()
def test():
    """This command runs the tests within docker and then exits. """
    command = "docker build -t strider-testing -f Dockerfile.test . && docker run strider-testing"
    print_green(command)
    os.system(command)


if __name__ == '__main__':
    main()
