---
title: "Helloworld OpenGL"
date: 2020-03-01T19:27:40+03:00
toc: false
images:
tags:
  - cmake
  - cpp
  - opengl
  - sdl2
---

In this post, we will create a simple application that initializes OpenGL and clears the buffer to red.

### What we will learn

- Create a Simple SDL2 Window.
- Create an OpenGL Context.
- Clear OpenGL Buffer.

### Table of Contents

- [Introduction](#introduction)
- [Create SDL2 Window](#window)
- [Create OpenGL Context](#context)
- [Main Rendering Loop](#rendering_loop)
- [Quit Event](#event)
- [Clear Buffer](#clear)
- [Cleanup](#cleanup)
- [Conclusion](#conclusion)

![Final Result][FinalResult]

---

### Introduction: <a name="introduction"/>

> OpenGL® is the most widely adopted 2D and 3D graphics API in the industry, bringing thousands of applications to a wide variety of computer platforms. [Source][khronos]

OpenGL is a **cross-platform 3D graphics API** that helps you create applications using graphics hardware. The fundamental concept of computer graphics is converting 3D data into 2D data — essentially, converting 3D models into 2D images.

We need a window to draw rendered images on. While native APIs (e.g., Windows API or X11) can be used to create windows, a cross-platform framework like **SDL2** simplifies things.

> Simple DirectMedia Layer (SDL) is a cross-platform development library designed to provide low-level access to audio, keyboard, mouse, joystick, and graphics hardware via OpenGL and Direct3D. [Source][SDL2]

---

### Create SDL2 Window: <a name="window"/>

First, we need to initialize SDL2 before making any function calls. SDL2 consists of many subsystems, but for now, we only need to initialize the **Video** system. [Source][SDL_Systems]

```cpp
if (SDL_Init(SDL_INIT_VIDEO) != 0) {
    std::cerr << "Cannot initialize SDL2\n";
    return EXIT_FAILURE;
}
```

Now, let's create an SDL2 window. [Source][SDL_Window]

```cpp
const auto width = 640U;
const auto height = 480U;
auto pWindow = SDL_CreateWindow("HelloWorld!", SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED, width, height, SDL_WINDOW_OPENGL);
if (pWindow == nullptr) {
    std::cerr << "Cannot create SDL2 window\n";
    return EXIT_FAILURE;
}
```

---

### Create OpenGL Context: <a name="context"/>

Next, we need to create an OpenGL context linked to our SDL2 window.

```cpp
SDL_GLContext context = SDL_GL_CreateContext(pWindow);
```

For now, this single line is enough.

---

### Main Rendering Loop: <a name="rendering_loop"/>

To keep the application running, we create a **main rendering loop**.

```cpp
bool bRunning = true;
while (bRunning) {
}
```

At this stage, the application enters an **infinite loop** and cannot quit. We need to handle user events to quit the application.

---

### Quit Event: <a name="event"/>

We must pull SDL events from the queue and handle the quit event.

```cpp
SDL_Event event;
while (SDL_PollEvent(&event) > 0) {
    switch (event.type) {
    case SDL_QUIT:
        bRunning = false;
        break;
    }
}
```

---

### Clear Buffer: <a name="clear"/>

Now, let's clear the OpenGL buffer. We'll set the clear color to **red** and swap the buffers to display the result.

```cpp
//           R, G, B, A
glClearColor(1, 0, 0, 1);  
glClear(GL_COLOR_BUFFER_BIT);
SDL_GL_SwapWindow(pWindow);
```

---

### Cleanup: <a name="cleanup"/>

Finally, we need to clean up the resources by deleting the OpenGL context and SDL2 window, then quitting SDL2.

```cpp
SDL_GL_DeleteContext(context);
SDL_DestroyWindow(pWindow);
SDL_Quit();
```

---

### Conclusion <a name="conclusion"/>

In this post, we’ve built a simple OpenGL application using SDL2 to create a window, establish an OpenGL context, and clear the buffer to red. We walked through the essential steps of setting up an SDL2 window, initializing OpenGL, handling events, and performing buffer clearing, which forms the foundation for more advanced OpenGL programming.

This is just the beginning — from here, you can explore more complex rendering techniques, shaders, and 3D models. With the basics of OpenGL and SDL2 in place, you now have the groundwork to develop your own graphics applications. Happy coding!

You can find the source code on GitHub: [SourceCode][SourceCode].

[FinalResult]: /images/clear-opengl-buffer/clear-opengl-buffer.png  
[khronos]: https://www.khronos.org/opengl/  
[SDL2]: https://wiki.libsdl.org/Introduction  
[SDL_Systems]: https://wiki.libsdl.org/SDL_Init  
[SDL_Window]: https://wiki.libsdl.org/SDL_CreateWindow  
[SourceCode]: https://github.com/wow2006/GraphicsDemos/blob/master/opengl/fixedPipelineOpenGLSDL.cpp