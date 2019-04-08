from setuptools import setup

setup(name='exmap',
      version='0.1',
      description='Explainable maps',
      url='http://github.com/boyanangelov/test',
      author='Boyan Angelov',
      author_email='boyan.angelov@gmail.com',
      license='MIT',
      packages=['exmap'],
      install_requires=['pandas',
                'xgboost',
                'scikit-learn',
                'lime',
                'folium',
                'tqdm'],
      zip_safe=False)
