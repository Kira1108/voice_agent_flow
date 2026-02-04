from setuptools import setup
from setuptools import find_packages

def read_requirements():
    with open('requirements.txt', 'r') as file:
        return [line.strip() for line in file if line.strip() and not line.startswith('#')]

setup(name='voice_agent_flow',
      version='0.0.1',
      description='Build llm agent for voice applications easily.',
      author='The fastest man alive.',
      packages=find_packages(),
      install_requires=read_requirements(),
      )