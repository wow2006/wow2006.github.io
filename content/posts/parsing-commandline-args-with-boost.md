---
title: "Parsing Command-Line Arguments in C++: A Guide with Boost"
date: 2019-08-15T19:27:40+03:00
toc: false
images:
tags:
  - cmake
  - cpp
  - boost
  - vcpkg
---

Command-line argument parsing is an essential feature for many applications. It enables you to pass information to your program during execution without hardcoding it into your code. This guide introduces how to parse command-line arguments in C++ starting with a naive example, and then explores a more sophisticated approach using Boost’s `program-options` library.

### Table of Contents

- [Introduction](#introduction)
- [Naive Command-Line Parsing Example](#naive)
- [Introduction to Boost `program-options`](#boost)
- [Boost `program-options` Example](#boost-example)
- [Advanced Boost `program-options` Example](#adv-boost-example)

---

### 1. Introduction<a name="introduction"/>

In C++, command-line arguments are passed to the `main` function via `argc` and `argv`:

- **`argc`**: The number of arguments passed to the program (including the program's name). It’s always greater than or equal to 1.
- **`argv`**: An array of C-style strings (`char*`) representing the command-line arguments. `argv[0]` is typically the program’s name or full path.

### 2. Naive Command-Line Parsing Example<a name="naive"/>

Let’s start with a basic example that checks the number of arguments passed and extracts two file names: an input file and an output file.

```cpp
#include <iostream>

int main(int argc, char* argv[]) {
  if (argc != 3) {
    std::cerr << "Usage:\n\t" << argv[0] << " input_file output_file\n";
    return EXIT_FAILURE;
  }

  const std::string inputFileName  = argv[1];
  const std::string outputFileName = argv[2];

  std::cout << inputFileName << ", " << outputFileName << '\n';

  return EXIT_SUCCESS;
}
```

This is a simple and direct approach, but it lacks the flexibility and features of more advanced argument parsers. For example, it doesn't handle arguments with dashes (e.g., `--help`), optional arguments, or argument validation.

### 3. Introduction to Boost `program-options`<a name="boost"/>

The Boost `program-options` library offers a powerful and flexible way to parse command-line arguments in C++. It allows you to:

- Use both single-dash (`-i`) and double-dash (`--input`) arguments.
- Specify required arguments.
- Automatically validate argument types.
- Group related arguments together for better organization.

While using Boost may feel like overkill for small programs, it provides a lot of useful functionality that can save time and make your code more robust in larger projects.

#### Installing Boost `program-options`:

- **Ubuntu**:

  ```bash
  sudo apt install libboost-program-options-dev -y
  ```

- **Arch Linux**:

  ```bash
  sudo pacman -S boost
  ```

- **Windows (via vcpkg)**:

  ```bash
  vcpkg install boost:x86-windows
  ```

### 4. Boost `program-options` Example<a name="boost-example"/>

Let’s modify our naive example using Boost `program-options` to parse input and output file arguments.

#### CMake Configuration

Here’s the `CMakeLists.txt` for building the project with Boost:

```cmake
cmake_minimum_required(VERSION 3.9)
project(naive-boost-example CXX)

find_package(Boost COMPONENTS program_options REQUIRED)

add_executable(
  naive-boost-example
  naive-boost-example.cpp
)

target_link_libraries(
  naive-boost-example
  PUBLIC
  Boost::program_options
)
```

#### Boost Example Code

```cpp
#include <iostream>

#include <boost/program_options.hpp>

int main(int argc, char* argv[]) {
  std::string input, output;
  using namespace boost::program_options;

  try {
    options_description desc{"Options"};
    desc.add_options()
      ("help,h", "Help screen")
      ("input,i", value<std::string>(&input)->required(), "Input file")
      ("output,o", value<std::string>(&output)->required(), "Output file");

    variables_map vm;
    store(parse_command_line(argc, argv, desc), vm);

    if (vm.count("help")) {
      std::cout << desc << '\n';
      return EXIT_SUCCESS;
    }

    notify(vm); // Validates the required options
  } catch (const error &ex) {
    std::cerr << ex.what() << '\n';
    return EXIT_FAILURE;
  }

  std::cout << input << ", " << output << '\n';
  return EXIT_SUCCESS;
}
```

#### Key Points

- `options_description` allows you to define the expected command-line options.
- `variables_map` stores the parsed options.
- The `notify()` function ensures that all required options are provided, and throws an error if not.

### 5. Advanced Boost `program-options` Example<a name="adv-boost-example"/>

Now let’s take it one step further by adding argument grouping. For example, you might want to separate arguments for image processing and general options.

#### Advanced Boost Example Code

```cpp
#include <iostream>

#include <boost/program_options.hpp>

int main(int argc, char* argv[]) {
  std::string input, output, calibration;
  using namespace boost::program_options;

  try {
    options_description general_desc{"General options"};
    general_desc.add_options()
      ("calib,c", value<std::string>(&calibration)->required(), "Calibration file");

    options_description image_desc{"Image options"};
    image_desc.add_options()
      ("input,i", value<std::string>(&input)->required(), "Input file")
      ("output,o", value<std::string>(&output)->required(), "Output file");

    options_description all_desc{"All options"};
    all_desc.add(general_desc).add(image_desc);
    all_desc.add_options()
      ("help,h", "Help screen")
      ("help-module", value<std::string>(), "Help for specific module");

    variables_map vm;
    store(parse_command_line(argc, argv, all_desc), vm);

    if (vm.count("help")) {
      std::cout << all_desc << '\n';
      return EXIT_SUCCESS;
    }

    if (vm.count("help-module")) {
      const auto module = vm["help-module"].as<std::string>();
      if (module == "general") {
        std::cout << general_desc << '\n';
      } else if (module == "image") {
        std::cout << image_desc << '\n';
      }
      return EXIT_SUCCESS;
    }

    notify(vm); // Validate the required options
  } catch (const error &ex) {
    std::cerr << ex.what() << '\n';
    return EXIT_FAILURE;
  }

  std::cout << input << ", " << output << '\n';
  return EXIT_SUCCESS;
}
```

---

### Conclusion

In this guide, we started with a simple way to handle command-line arguments and evolved it into a robust and flexible solution using Boost `program-options`. This library provides extensive features that make argument parsing easier and more maintainable, especially for larger projects with complex options.
