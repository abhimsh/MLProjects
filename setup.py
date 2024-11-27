from setuptools import setup, find_packages
from typing import List

E_DOT = "-e ."

def get_all_requirements() -> List[str]:
    """
    This method will get a list of all required python packgages from requirements.txt
    """
    with open("requirements.txt") as file_obj:
        requirements = file_obj.readlines()

    requirements = [each_req.replace("\n", "") for each_req in requirements if each_req.strip("\n") != E_DOT]
    return requirements


setup(
    name="mlprojects",
    version="0.0.1",
    author="Abhimsh",
    author_email="abhimshedu@gmail.com",
    packages=find_packages(),
    install_requires=get_all_requirements()
)