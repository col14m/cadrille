from setuptools import setup

setup(
    name="vllm_register_cadrille",
    version="0.1",
    packages=["cadrille_plugin"],
    package_dir={"cadrille_plugin": "cadrille_plugin"},
    entry_points={
        "vllm.general_plugins": [
            "cadrille = cadrille_plugin:register",
        ]
    },
)