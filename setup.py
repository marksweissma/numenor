from setuptools import setup, find_packages


REQUIRES = open('package_requirements.txt', 'r').read()


setup(name='numenor',
      version='0.0.1',
      author='Mark Weiss',
      author_email='mark.s.weiss.ma@gmail.com',
      description=open('README.md'),
      url='http://github.com/marksweissma/numenor',
      packages=find_packages(),
      install_requires=REQUIRES,
      summary='ml infra utils',
      license='MIT',
      zip_safe=False
      )
