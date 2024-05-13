from setuptools import setup

setup(
    name='MagicTools',
    version='0.1.0',
    description='A python package for training models using DistributedDataParallel framework',
    url='https://github.com/ZVengin/MagicTools.git',
    author='Zhong Wenjie',
    author_email='wenjiezhong717@gmail.com',
    license='MIT License',
    packages=['MagicTools'],
    install_requires=['transformers',
                      'tqdm',
                      'wandb',
                      'torch',
                      'numpy',
                      'datasets',
                      'pandas',
                      'nltk'
                      ],

    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 3',
    ],
)
