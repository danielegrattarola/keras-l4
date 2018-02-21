from setuptools import setup

setup(
    name='keras_l4',
    version='1.0',
    packages=['keras_l4'],
    install_requires=['keras'],  # Also requires tensorflow, but I don't want to mess up people's installs
    url='https://github.com/danielegrattarola/keras-l4',
    license='MIT',
    author='Daniele  Grattarola',
    author_email='daniele.grattarola@gmail.com',
    description='A Keras implementation of L4 stepsize adaptation schemeproposed by Rolinek, Martius (2018).'
)
