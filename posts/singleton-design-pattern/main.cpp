#include <chrono>
#include <iostream>
#include <memory>

class Singleton {
public:
  static Singleton *Instance();
  ~Singleton() = default;
  std::uint32_t value = 1u;

private:
  Singleton() = default;

private:
  static Singleton *_instance;
};

Singleton *Singleton::_instance = nullptr;

Singleton *Singleton::Instance() {
  static char data[sizeof(Singleton)];
  if (_instance == nullptr) {
    _instance = new (data) Singleton;
  }
  return _instance;
}

class Singleton2 {
public:
  static Singleton2 *Instance();

  ~Singleton2() = default;

  std::uint32_t value = 1u;

private:
  Singleton2() = default;
};

Singleton2 *Singleton2::Instance() {
  static Singleton2 s;
  return &s;
}

class Singleton3 {
public:
  static Singleton3 *Instance();

  ~Singleton3() = default;

  std::uint32_t value = 1u;

  static Singleton3 s;
private:
  Singleton3() = default;
};

Singleton3 Singleton3::s;

Singleton3 *Singleton3::Instance() {
  return &s;
}

constexpr std::uint64_t count = 5'000'000'000;

int main(int argc, char *argv[]) {
  {
    std::uint64_t sum = 0u;

    const auto t1 = std::chrono::high_resolution_clock::now();
    for (std::uint64_t i = 0; i < count; ++i) {
      sum += Singleton::Instance()->value;
    }
    const auto t2 = std::chrono::high_resolution_clock::now();

    std::cout << sum << '\n';

    std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1)
                     .count()
              << '\n';
  }

  {
    std::uint64_t sum = 0u;

    const auto t1 = std::chrono::high_resolution_clock::now();
    for (std::uint64_t i = 0; i < count; ++i) {
      sum += Singleton2::Instance()->value;
    }
    const auto t2 = std::chrono::high_resolution_clock::now();

    std::cout << sum << '\n';

    std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1)
                     .count()
              << '\n';
  }

  {
    std::uint64_t sum = 0u;

    const auto t1 = std::chrono::high_resolution_clock::now();
    for (std::uint64_t i = 0; i < count; ++i) {
      sum += Singleton3::Instance()->value;
    }
    const auto t2 = std::chrono::high_resolution_clock::now();

    std::cout << sum << '\n';

    std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1)
                     .count()
              << '\n';
  }

  return 0;
}
