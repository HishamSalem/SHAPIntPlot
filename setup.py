from setuptools import setup, find_packages

setup(
    name='my_shap_package',
    version='0.1.0',
    description='A package for SHAP value interaction analysis and visualization',
    author='Your Name',
    author_email='your.email@example.com',
    url='https://github.com/HishamSalem/SHAP_Interactions',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'matplotlib',
        'networkx',
        'scipy',
        'shap',
        'xgboost',
        'scikit-learn'
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)
