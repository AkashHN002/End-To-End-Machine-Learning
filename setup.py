from setuptools import find_packages,  setup
from typing import List


hypen_E_dot = '-e .'
def get_requirements(file_path:str) ->List[str]:
    '''
    this functioin will return the list of requirements
    '''
    requirements = []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace('\n', '') for req in requirements]

        if hypen_E_dot in requirements:
            requirements.remove(hypen_E_dot)

    return requirements

setup(
    name = 'mlproject',
    version='0.0.1',
    author='Akash H N',
    author_email='akashhn06@gmail.com',
    packages= find_packages(),
    install_requires = get_requirements('requirements.txt')

)