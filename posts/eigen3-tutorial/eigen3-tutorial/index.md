<!--
.. title: Eigen3 Tutorial
.. slug: eigen3-tutorial
.. date: 2019-08-14 02:02:18 UTC+02:00
.. tags: eigen, cpp, cmake
.. category:
.. link:
.. description: Eigen tutorial
.. type: text
-->

####Content:
1. [Install](#install)
2. [CMake](#cmake)
3. [SimpleExample](#simple)

####Install Eigen3:<a name="install"/>
- Ubuntu 18.04:
```bash
sudo apt install libeigen3-dev -y
```

- Arch Linux:
```bash
sudo pacman -S eigen --noconfirm
```

- Windows using vcpkg:
```bash
vcpkg install eigen3
```

####Use Eigen3 with CMake:<a name="cmake"/>
- Create project directory
```bash
mkdir eigen3_tutorial/{src,build} -p  # Create project root directory
cd eigen3_tutorial/src                # Change directory to src
touch CMakeLists.txt main.cpp         # Create empty CMakeLists.txt and main.cpp
```

- Create `CMakeLists.txt`
```CMakeLists.txt
cmake_minimum_required(VERSION 3.9)
project(eigen3_tutorial CXX)

find_package(Eigen3 REQUIRED)

add_executable(
  eigen3_tutorial
  main.cpp
)

target_link_libraries(
  eigen3_tutorial
  PRIVATE
  Eigen3::Eigen
)
```

- Create `main.cpp`
```cpp
#include <iostream>

int main(int argc, char* argv[]) {
  return 0;
}
```

####Create simple example:<a name="simple"/>
```cpp
// STL
#include <iostream>
// Eigen3
#include <Eigen/Core> // TODO(Hussein)

int main(int argc, char* argv[]) {
  Eigen::Vector3d vec3; // TODO(Hussein)
  vec3 << 1, 2, 3;      // TODO(Hussein)

  std::cout << vec3.transpose() << '\n';
  return 0;
}

```

