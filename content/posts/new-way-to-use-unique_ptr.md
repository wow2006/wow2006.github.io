---
title: "New way to use unique_ptr"
date: 2024-11-16T16:35:38+02:00
toc: false
images:
tags:
  - cpp
  - smart_pointers
---

In the previous blog post, we implemented a simple `unique_ptr` class to demonstrate how its internal mechanics work. One crucial aspect of `unique_ptr` is the deleter, which I briefly mentioned earlier. In this post, we will focus on the deleter in detail. To begin, let's review its standard definition [1]

```cpp
template<
    class T,
    class Deleter = std::default_delete<T>
> class unique_ptr;
```

The deleter is a callable entity that operates on a pointer of the specified template type. By default, it uses the `delete` operator to free the allocated memory. However, the key feature of `unique_ptr` is its flexibility, allowing users to replace the default deleter with a custom one.

For instance, consider a scenario where a codebase manages files using `fopen` and `fclose` instead of `std::fstream`. Suppose the code owner prefers this approach for some unspecified reason:

```cpp
FILE* handle = fopen("sample", "r");
if (nullptr == handle) {
    return -1;
}
// Perform operations
fclose(handle);
```

By leveraging the powerful RAII (Resource Acquisition Is Initialisation) technique, developers can use `unique_ptr` to manage resources beyond memory, such as sockets, files, or database connections, ensuring appropriate cleanup at the end of the scope. 

Here’s how to adapt the previous example to use `unique_ptr`:

```cpp
#include <cstdio>
#include <memory>

int main() {
    using Handler = std::unique_ptr<FILE, decltype(&fclose)>;
    auto fileHandler = Handler(fopen("sample", "r"), &fclose);
    if (!fileHandler) { // utilises the `operator bool`
        return -1;
    }
    // Perform operations
    return 0;
}
```

### Using a Lambda to Execute a Function at Scope End

`unique_ptr` can also invoke a function when it goes out of scope, which is particularly useful for non-memory resource management. Here’s an example:

```cpp
#include <iostream>
#include <memory>

void end_of_scope(int* final_count) {
    std::cout << "Final count: " << *final_count << "\n";
}

int main() {
    using ScopeFunc = std::unique_ptr<int, decltype(&end_of_scope)>;
    int count = 100;
    ScopeFunc scope{&count, &end_of_scope};
    return 0;
}
```

### Managing Dynamically Allocated Arrays

`unique_ptr` supports dynamically allocated arrays, though it's worth noting that using `std::string` or `std::vector` is often more practical. However, in some embedded environments, using these containers may be restricted. Here’s an example of managing a dynamically allocated array with `unique_ptr`:

```cpp
#include <iostream>
#include <memory>

int main() {
    std::unique_ptr<char[]> message{new char[]{"HelloWorld!"}};
    std::cout << message.get() << '\n';
    return 0;
}
```

While alternatives like `std::string` or `std::vector` are generally preferred for such tasks, `unique_ptr` with arrays remains a valid option, especially in environments with limited library support.

In summary, `unique_ptr` is a versatile tool for managing dynamically allocated resources, whether they are memory, files, or other types of resources, thanks to its customisable deleter.

[1]: https://en.cppreference.com/w/cpp/memory/unique_ptr "unique_ptr"
