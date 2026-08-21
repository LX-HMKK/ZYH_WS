from setuptools import setup, find_packages

setup(
	name="gloria-msdk",
	version="0.0.1",
	description="Gloria-M 夹爪 Python SDK",
	packages=find_packages(),
	install_requires=["pyserial>=3.5"],
	python_requires=">=3.9",
)
