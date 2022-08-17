from setuptools import setup, find_packages


setup(
    name='dcpsfrgan',
    version='0.1dev',
    packages=find_packages(),
    license='',
    python_requires=">=3.8.*",
    install_requires=[
	torch,
	torchvision,
	tensorflow,
	tensorboard,
	tensorboardX,
	opencv-python,
	dlib,
	scikit-image,
	scipy,
	tqdm,
	imgaug,
    ],
    include_package_data=True,
    package_data={'': ['*.yaml']},
    long_description=open('README.md', encoding='utf-8').read(),
    long_description_content_type='text/markdown'
)
