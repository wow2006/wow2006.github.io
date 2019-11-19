<!--
.. title: Singleton Design Pattern
.. slug: singleton-design-pattern
.. date: 2019-11-19 17:33:51 UTC+02:00
.. tags:
.. category: desgin-pattern modern-cpp
.. link:
.. description: Implement Singletone
.. type: text
-->

####Content:
1. [Idea](#idea)
2. [UML](#uml)
3. [Simple Implementation](#simple-impl)
4. [Other Implementation](#other-impl)
4. [References](#other-impl)

#####Idea:<a name="idea" />
I hate this design so I will start with it. Singleton is one of the structural patterns. It is responsible for creating just one instance of a class.

> Ensure a class only has one instance, and provide a global point of access to it.
>
> Gang of four


#####UML:<a name="uml" />


#####Simple Implementation:<a name="simple-impl" />
c++98 Implementation
```cpp
class Singleton {
public:
  static Singleton* Instance();
protected:
  Singleton();
private:
  static Singleton* _instance;
};

Singleton* Singleton::_instance = 0;

Singleton::Singleton() {}

Singleton* Singleton::Instance() {
  if(_instance == 0) {
    _instance = new Singleton;
  }
  return _instance;
}
```

c++11 Implementation
```cpp
#include <memory>

class Singleton {
public:
  static Singleton* Instance();
  ~Singleton() = default;
private:
  Singleton() = default;
private:
  static std::unique_ptr<Singleton> _instance;
};

std::unique_ptr<Singleton> Singleton::_instance;

Singleton* Singleton::Instance() {
  if(!_instance) {
    _instance = std::unique_ptr<Singleton>();
  }
  return _instance.get();
}
```

#####Other Implementation:<a name="other-impl" />
c++11 Implementation
```cpp
class Singleton {
public:
  static Singleton* Instance();

private:
  Singleton() = default;

};

Singleton* Singleton::Instance() {
  static Singleton sInstance;
  return &sInstance;
}
```

#####References:<a name="references">
- Design Patterns: Elements of Reusable Object-Oriented Software (1994)
- Design Patterns in Modern C++ - Dmitri Nesteruk
- Head First: Design Patterns
- Game Programming: Patterns
- [Ep 2 Cost of Using Statics](https://www.youtube.com/watch?v=B3WWsKFePiM)
