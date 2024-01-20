"""Setup file for SRI Test Harness package."""
from setuptools import setup

with open("README.md", encoding="utf-8") as readme_file:
    readme = readme_file.read()

setup(
    name="strider",
    version="4.5.1",
    author="Max Wang",
    author_email="max@covar.com",
    url="https://github.com/ranking-agent/strider",
    description="Strider",
    long_description_content_type="text/markdown",
    long_description=readme,
    packages=["strider"],
    include_package_data=True,
    zip_safe=False,
    license="MIT",
    python_requires=">=3.9",
)
