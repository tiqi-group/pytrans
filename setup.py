from setuptools import setup, find_packages

VERSION = '1.9.0'
BASE_CVS_URL = 'https://gitlab.phys.ethz.ch/tiqi-projects/pytrans.git'

setup(
    name='pytrans',
    packages=find_packages(),
    version=VERSION,
    author='tiqi',
    author_email='',
    description='Transport waveform-related generation and analysis utilities',
    install_requires=[x.strip() for x in open('requirements.txt').readlines()],
    url=BASE_CVS_URL,
    test_suite='tests',
    keywords=[],
    classifiers=[
        "Development Status :: 1 - Planning",
        "Intended Audience :: Developers",
        "Programming Language :: Python",
        "Operating System :: OS Independent",
    ],
)
