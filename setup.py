"""Setup file for strider package."""
from setuptools import setup

setup(
    name='strider',
    version='0.1.0-dev0',
    author='Patrick Wang',
    author_email='patrick@covar.com',
    url='https://github.com/TranslatorIIPrototypes/strider',
    description='Strider (Ranking Agent)',
    packages=['strider'],
    include_package_data=True,
    zip_safe=False,
    license='MIT',
    python_requires='>=3.8',
)
