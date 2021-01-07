
def pytest_addoption(parser):
    """ Add option for long running tests """
    parser.addoption('--longrun', action='store_true', dest="longrun",
                     default=False, help="enable longrundecorated tests")


def pytest_configure(config):
    """ Set option to false by default """
    if not config.option.longrun:
        setattr(config.option, 'markexpr', 'not longrun')
