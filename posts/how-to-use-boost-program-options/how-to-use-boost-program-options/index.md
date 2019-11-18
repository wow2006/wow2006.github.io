<!--
.. title: How to use Boost program options
.. slug: how-to-use-boost-program-options
.. date: 2019-08-15 18:16:40 UTC+02:00
.. tags: cpp,boost,program-options
.. category:
.. link:
.. description: Introdcution to Boost program options.
.. type: text
-->

##### Content:
- [Introduction](#introduction)
- [Naive Example](#naive)
- [Intruction to Boost `program-options`](#boost)
- [Boost Example](#boost-example)
- [Advanced Boost Example](#adv-boost-example)

##### Introduction: <a name="introduction"/>
I want to pass arguments to a program. so, I will check `argc` and `argv` for input.

###### side-note:
- `argc` is a count of arguments passed to our program. It is always greater than 1.
- `argv` is an array of `char*` of arguments. First `argv[0]` is the full path for our program.

##### Naive Example: <a name="naive"/>
I will show simple example without `Boost`.
```cpp
// STL
#include <iostream>

int main(int argc, char* argv[]) {
  if(argc != 3) {
    std::cerr << "Usage:\n\t" << argv[0] << " input_file output_file\n";
    return EXIT_FAILURE;
  }

  const std::string inputFileName  = argv[1];
  const std::string outputFileName = argv[2];

  std::cout << inputFileName << ", " << outputFileName << '\n';

  return EXIT_SUCCESS;
}
```

Still, It will take some effort to make it match with Unix [standard](http://tldp.org/LDP/abs/html/standard-options.html).

##### Intruction to Boost `program-options`: <a name="boost"/>

Program options are a library for check and parse program options.
For a simple application, It will look like overkilling. Still, It will
provide us with a lot of free features.

- We can use dash and double dash argument.
- We can specify which argument is required.
- We can specify a variable type.
- We can group arguments.

`That all I use`

###### Installation:

- Ubuntu:
`sudo apt install libboost-program-options-dev -y`
or for full boost
`sudo apt install libboost-all-dev -y`
- Arch Linux:
`sudo pacman -S boost`
- Windows `vcpkg`:
`vcpkg install boost:x86-windows`

##### Boost Example: <a name="boost-example"/>

```CMakeLists.txt
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

```cpp
// STL
#include <iostream>
// Boost
#include <boost/program_options.hpp>

int main(int argc, char* argv[]) {
  std::string input, output;
  using namespace boost::program_options;
  try {
    options_description desc{"Options"};
    desc.add_options()("help,h", "Help screen")
    ("input,i", value<std::string>(&input)->required(), "Input file")
    ("output,o", value<std::string>(&output)->required(), "Input output");

    variables_map vm;
    store(parse_command_line(argc, argv, desc), vm);

    if (vm.count("help")) {
      std::cout << desc << '\n';
      return EXIT_SUCCESS;
    }

    notify(vm);
  } catch (const error &ex) {
    std::cerr << ex.what() << '\n';
    return EXIT_FAILURE;
  }

  std::cout << input << ", " << output << '\n';
  return EXIT_SUCCESS;
}
```

##### Advanced Boost Example: <a name="adv-boost-example"/>

Now we need to use grouping. I want to make two options group.
First for images and the second for general options.
We can also add description for each one. We can add `help-module` and check input.

```cpp
// STL
#include <iostream>
// Boost
#include <boost/program_options.hpp>

int main(int argc, char* argv[]) {
  std::string input, output;
  std::string calibration;
  using namespace boost::program_options;
  try {
    options_description all_desc{"All options"};
    all_desc.add_options()
      ("help,h", "Help screen")
      ("help-module", value<std::string>(), "Help for every module");

    options_description desc{"general options"};
    desc.add_options()
    ("calib,c", value<std::string>(&calibration)->required(), "Calibration fil");

    options_description image_desc{"image options"};
    image_desc.add_options()
    ("input,i", value<std::string>(&input)->required(), "Input file")
    ("output,o", value<std::string>(&output)->required(), "Input output");

    all_desc.add(desc).add(image_desc);

    variables_map vm;
    store(parse_command_line(argc, argv, all_desc), vm);

    if (vm.count("help")) {
      std::cout << all_desc << '\n';
      return EXIT_SUCCESS;
    }

    if (vm.count("help-module")) {
      const auto module = vm["help-module"].as<std::string>();
      if(module.empty()) {
        std::cerr << "SHIT\n";
      }
      std::cout << "IN " << module << '\n';
      if(module == "general") {
        std::cout << desc << '\n';
      } else if(module == "image") {
        std::cout << image_desc << '\n';
      }
      return EXIT_SUCCESS;
    }

    notify(vm);
  } catch (const error &ex) {
    std::cerr << ex.what() << '\n';
    return EXIT_FAILURE;
  }

  std::cout << input << ", " << output << '\n';
  return EXIT_SUCCESS;
}
```

