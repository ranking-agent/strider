import os

import pytest


@pytest.fixture(scope="function", autouse=True)
def set_working_directory(request):
    """
    Allow us to load data files from the test directory
    """
    os.chdir(request.fspath.dirname)
