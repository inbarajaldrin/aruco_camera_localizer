from setuptools import find_packages, setup
from glob import glob

package_name = 'aruco_camera_localizer'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/STL', glob('STL/*.STL')),
    ],
    install_requires=['setuptools', 'cv2', 'scipy'],
    zip_safe=True,
    maintainer='max',
    maintainer_email='max@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'localize = aruco_camera_localizer.merged_localization:main'
        ],
    },
)
