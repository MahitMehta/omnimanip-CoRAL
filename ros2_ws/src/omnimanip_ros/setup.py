from setuptools import setup

package_name = "omnimanip_ros"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Mahit Mehta",
    maintainer_email="mahit.py@gmail.com",
    description="OmniManip stages as ROS 2 services",
    license="MIT",
    entry_points={
        "console_scripts": [
            "bridge = omnimanip_ros.bridge_node:main",
            "run_task = omnimanip_ros.run_task:main",
            "execute = omnimanip_ros.execute_node:main",
        ],
    },
)
