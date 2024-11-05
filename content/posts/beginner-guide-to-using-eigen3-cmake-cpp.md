---
title: "A Beginner's Guide to Using Eigen3 with CMake in C++"
date: 2019-08-14T19:27:40+03:00
toc: false
images:
tags:
  - cmake
  - cpp
  - eigen
  - vcpkg
---

Eigen3 is a powerful and versatile C++ library used for linear algebra, making it a great choice for scientific computing, 3D graphics, or any application involving matrices and vectors. This guide will walk you through how to install Eigen3, set up a simple project using CMake, and write a basic Eigen3 example in C++.

### Table of Contents

1. [Installing Eigen3](#install)
2. [Setting Up CMake](#cmake)
3. [Creating a Simple Eigen3 Example](#example)

By the end of this post, you'll have a small project that compiles and runs a basic example using Eigen3. This post should take about 10 minutes to read and follow along.

---

### 1. Installing Eigen3<a name="install"/>

Before diving into coding, you need to have Eigen3 installed on your system. Here's how to do it on different platforms:

#### Ubuntu 18.04

On Ubuntu, installing Eigen3 is straightforward with the following command:

```bash
sudo apt install libeigen3-dev -y
```

#### Arch Linux

If you're using Arch Linux, you can install it via `pacman`:

```bash
sudo pacman -S eigen --noconfirm
```

#### Windows using vcpkg

For Windows users, we recommend using `vcpkg` to install Eigen3:

```bash
vcpkg install eigen3
```

Once installed, you’re ready to set up your project using CMake.

---

### 2. Setting Up CMake<a name="cmake"/>

Now that Eigen3 is installed, let’s create a simple project structure and set up CMake to use Eigen3.

#### Step 1: Create the Project Directory

Start by creating a new project folder and its subdirectories:

```bash
mkdir eigen3_tutorial/{src,build} -p
cd eigen3_tutorial/src
touch CMakeLists.txt main.cpp
```

This will create a folder structure with a `src` directory for your source code and a `build` directory where CMake will generate the necessary files for compiling your project.

#### Step 2: Create the `CMakeLists.txt`

In the `src` directory, open the `CMakeLists.txt` file and add the following content:

```cmake
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

This CMake configuration will find the Eigen3 library and link it to your project. It ensures that Eigen3 is properly included in your build process.

#### Step 3: Create the `main.cpp`

Now let’s create a simple `main.cpp` file:

```cpp
int main(int argc, char* argv[]) {
  return 0;
}
```

This empty main function serves as a placeholder. Next, we’ll add a small example using Eigen3.

---

### 3. Creating a Simple Eigen3 Example<a name="example"/>

Let’s modify the `main.cpp` to include a basic example with Eigen3. In this example, we’ll create a vector and print its transpose to the console.

Update the `main.cpp` file with the following code:

```cpp
#include <iostream>

#include <Eigen/Core>

int main(int argc, char* argv[]) {
  Eigen::Vector3d vec3;
  vec3 << 1, 2, 3;

  std::cout << vec3.transpose() << '\n';
  return 0;
}
```

#### Explanation

- We include the `<Eigen/Core>` header to use Eigen’s core functionalities.
- We create a 3D vector (`Eigen::Vector3d`), assign the values 1, 2, and 3, and then print the transpose of this vector to the console.

#### Step 4: Build and Run the Project

Navigate to the `build` directory and run the following commands to compile and run the project:

```bash
cd ../build
cmake ../src
make
./eigen3_tutorial
```

If everything is set up correctly, you should see the following output:

```
1 2 3
```

---

### Conclusion

You’ve just installed Eigen3, set up a simple CMake project, and written a basic example in C++ using Eigen3. Eigen3 is incredibly powerful, and this guide serves as a foundation to get you started. You can now explore more complex operations and use Eigen3 in your projects.
