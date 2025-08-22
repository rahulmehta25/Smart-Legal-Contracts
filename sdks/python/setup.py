from setuptools import setup, find_packages

setup(
    name='api-marketplace-sdk',
    version='1.0.0',
    description='Official Python SDK for API Marketplace',
    author='API Marketplace Team',
    author_email='sdk-support@example.com',
    url='https://github.com/example/api-marketplace-sdk-python',
    packages=find_packages(),
    install_requires=[
        'requests>=2.25.0',
        'oauth2client>=4.1.3',
        'jsonschema>=3.2.0'
    ],
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10'
    ],
    python_requires='>=3.8',
    keywords='api marketplace sdk'
)