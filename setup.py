from setuptools import setup, find_packages

def load_requirements():
    with open("requirements.txt", "r", encoding="utf-8") as f:
        requirements = []
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                requirements.append(line)
    return requirements

setup(
    name="ml4investment",
    version="1.0.0",
    packages=find_packages(),
    install_requires=load_requirements(),
)