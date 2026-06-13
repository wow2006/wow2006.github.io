---
title: "Write a basic unique_ptr in cpp"
date: 2024-11-06T00:06:27+02:00
toc: false
images:
tags:
  - cpp
  - smart_pointers
---

Modern C++ encompasses post-C++11 standards, including C++11 itself. C++11 marked one of the most substantial updates in the language’s history, introducing a vast range of features. These updates are generally divided into two categories: language features and library features (STL).

One of the most impactful additions to the C++ Standard Library was smart pointers. Although smart pointers were initially introduced by the Boost library[1], they became part of the standard in C++11. Notably, auto_ptr was deprecated in C++11 and removed in C++17 due to its limitations and potential for misuse[2]. This blog will focus specifically on the implementation of unique_ptr.

Before diving into the implementation, we need to understand the why, how, and when of using unique_ptr. C++ is a language that provides developers with the freedom to manage memory manually. It offers two types of memory management: stack and heap. Stack memory is managed automatically by the C runtime, with allocation starting at a function’s scope and deallocation occurring when that scope ends. Conversely, heap memory is under the programmer’s control. To allocate memory on the heap, one can use malloc, calloc, or realloc from C, or in C++ we can use new and new[] operators.

`Hint: new is an operator not a function` [cppreference][3]

Since we are focusing on C++, we will adhere to the new operator for heap allocation in this discussion.

```cpp
#include <string>
// ...
std::string* name = new std::string;
delete name;
```

As you can see, the user is required to manually deallocate or free memory, which is highly error-prone. For instance, forgetting to free memory, returning from a function before deleting it, or encountering an exception can all lead to memory leaks and undefined behaviour.

This is where smart pointers shine. Smart pointers handle memory management for you, automatically deallocating memory when it is no longer needed. For example, consider replacing the previous code with:

```cpp
auto name = std::unique_ptr<std::string>(new std::string);
```

Here, name controls the heap-allocated memory, ensuring it is properly freed when name goes out of scope.

The principle of managing heap memory based on object lifetime is a C++ technique known as RAII (Resource Acquisition Is Initialisation). RAII ensures that resources, such as dynamic memory, are automatically cleaned up when their owning object goes out of scope.

Let’s now write a simple class to manage memory within a scope. We will call it `scope_ptr`.

```cpp
#include <cstdlib>
#include <string>

template<typename Type>
class scope_ptr {
public:
    using pointer = Type*;
    using element_type = Type;

    /**
     * @brief Default constructor creating an empty scope_ptr.
     */
    scope_ptr() noexcept = default;

    /**
     * @brief Constructor taking ownership of a raw pointer.
     *
     * @param ptr Raw pointer to manage.
     * @throws Nothing.
     */
    explicit scope_ptr(pointer ptr) noexcept
        : mPtr{ptr} {
        static_assert(!std::is_array_v<Type>, "Arrays are not supported");
    }

    // Disable copy construction and assignment
    scope_ptr(const scope_ptr&) = delete;
    scope_ptr& operator=(const scope_ptr&) = delete;

    // Disable move construction and assignment
    scope_ptr(scope_ptr&&) = delete;
    scope_ptr& operator=(scope_ptr&&) = delete;

    /**
     * @brief Destructor that ensures the managed object is deleted.
     */
    ~scope_ptr() {
        reset();
    }

    /**
     * @brief Get the raw pointer without transferring ownership.
     *
     * @return pointer The raw pointer being managed.
     */
    [[nodiscard]] pointer get() const noexcept {
        return mPtr;
    }

    /**
     * @brief Dereference operator.
     *
     * @return Type& Reference to the managed object.
     * @throws Nothing.
     * @pre The pointer must not be null.
     */
    [[nodiscard]] Type& operator*() const noexcept {
        assert(mPtr != nullptr && "Dereferencing null pointer");
        return *mPtr;
    }

    /**
     * @brief Member access operator.
     *
     * @return pointer The raw pointer for member access.
     * @throws Nothing.
     * @pre The pointer must not be null.
     */
    [[nodiscard]] pointer operator->() const noexcept {
        assert(mPtr != nullptr && "Accessing member through null pointer");
        return mPtr;
    }

    /**
     * @brief Boolean conversion operator.
     *
     * @return true If the pointer is non-null.
     * @return false If the pointer is null.
     */
    [[nodiscard]] explicit operator bool() const noexcept {
        return mPtr != nullptr;
    }

    /**
     * @brief Release ownership of the pointer without deleting it.
     *
     * @return pointer The previously managed pointer.
     */
    [[nodiscard]] pointer release() noexcept {
        pointer temp = mPtr;
        mPtr = nullptr;
        return temp;
    }

    /**
     * @brief Reset the managed pointer, deleting the current object if any.
     *
     * @param ptr New pointer to manage (defaults to nullptr).
     */
    void reset(pointer ptr = nullptr) noexcept {
        if (mPtr != ptr) {
            delete mPtr;
            mPtr = ptr;
        }
    }

    /**
     * @brief Swap the managed pointer with another scope_ptr.
     *
     * @param other The other scope_ptr to swap with.
     */
    void swap(scope_ptr& other) noexcept {
        std::swap(mPtr, other.mPtr);
    }

private:
    pointer mPtr = nullptr;  ///< The managed pointer
};

/**
 * @brief Example usage of scope_ptr.
 */
auto main(int argc, char* argv[]) -> int {
    // Create a scope_ptr managing a string
    scope_ptr<std::string> ptr{new std::string};

    // Use the managed string
    ptr->append("HelloWorld!");

    // Return the size - ptr will be automatically deleted
    return static_cast<int>(ptr->size());
}
```

This version disables both copy and move constructors, as well as assignment operators, to ensure unique ownership and prevent accidental misuse. Although this implementation is limited in functionality, it provides a good foundation for understanding how smart pointers work:

```cpp
#include <cassert>
#include <cstdlib>
#include <iostream>
#include <string>
#include <type_traits>

template <typename Type>
class unique_ptr final {
  // Ensure Type is not a reference type
  static_assert(!std::is_reference<Type>::value, "");
  // Ensure Type is not const-qualified
  static_assert(!std::is_const<Type>::value, "");

public:
  using Ptr = Type*;  ///< Alias for pointer to managed type
  using Ref = Type&;  ///< Alias for reference to managed type

  /**
   * @brief Default constructor initializing with nullptr
   */
  unique_ptr() = default;

  /**
   * @brief Constructor taking raw pointer ownership
   * @param ptr Raw pointer to manage
   */
  unique_ptr(Ptr ptr) : mPtr{ptr} {}

  // Disable copy operations to ensure unique ownership
  unique_ptr(const unique_ptr&) = delete;
  unique_ptr& operator=(const unique_ptr&) = delete;

  /**
   * @brief Move constructor
   * @param other Source unique_ptr to move from
   *
   * Transfers ownership from other to this instance
   */
  unique_ptr(unique_ptr&& other) { std::swap(other.mPtr, mPtr); }

  /**
   * @brief Move assignment operator
   * @param other Source unique_ptr to move from
   * @return Reference to this instance
   */
  unique_ptr& operator=(unique_ptr&& other) {
    std::swap(other.mPtr, mPtr);
    return *this;
  }

  /**
   * @brief Destructor that ensures managed resource is properly deleted
   */
  ~unique_ptr() { reset(); }

  /**
   * @brief Get the raw pointer
   * @return Raw pointer to managed object
   */
  Ptr get() const { return mPtr; }

  /**
   * @brief Boolean conversion operator
   * @return true if managing a non-null pointer
   */
  operator bool() const { return nullptr != mPtr; }

  /**
   * @brief Arrow operator for accessing member functions
   * @return Raw pointer to managed object
   */
  Ptr operator->() const { return mPtr; }

  /**
   * @brief Dereference operator
   * @return Reference to managed object
   * @throws Assertion failure if pointer is null
   */
  Ref operator*() const {
    assert(nullptr != mPtr);
    return *mPtr;
  }

  /**
   * @brief Releases ownership of managed pointer
   * @param ptr Optional new pointer to manage (defaults to nullptr)
   * @return Previously managed pointer
   */
  Ptr release(Ptr ptr = nullptr) {
    auto* temp = mPtr;
    mPtr = ptr;
    return temp;
  }

  /**
   * @brief Deletes currently managed object and sets pointer to nullptr
   */
  void reset() {
    if (nullptr != mPtr) {
      delete mPtr;
      mPtr = nullptr;
    }
  }

private:
  Ptr mPtr = nullptr;  ///< The managed pointer
};

/**
 * @brief Example usage of unique_ptr
 */
auto main(int argc, char *argv[]) -> int {
  unique_ptr<std::string> ptr{new std::string};
  ptr->append("HelloWorld!");

  std::cout << *ptr << '\n';

  return EXIT_SUCCESS;
}
```

This initial implementation serves as a foundation, but enhancing it with custom deleters and move semantics will help achieve the full power of modern C++ smart pointers. However, it lacks several features that make modern smart pointers robust.
Custom Deleter Support is missing here. A custom deleter is essential when working with non-standard memory resources, such as file handles or network connections.

[1]: https://www.boost.org/doc/libs/1_65_0/libs/smart_ptr/doc/html/smart_ptr.html "boost smart_ptr"
[2]: https://en.cppreference.com/w/cpp/memory/auto_ptr "cppreference auto_ptr"
[3]: https://en.cppreference.com/w/cpp/memory/new/operator_new  "cppreference operator_new"
