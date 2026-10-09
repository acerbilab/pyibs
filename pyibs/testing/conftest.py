"""pytest's configuration of the tests, also where they run from the
installed package, as ``pytest --pyargs pyibs``."""


def pytest_configure(config):
    config.addinivalue_line("markers", "integration: runs PyBADS or PyVBMC")
