<!--
.. title: Save OpenGL frame to image
.. slug: save-opengl-frame
.. date: 2020-01-02 02:02:18 UTC+02:00
.. tags: opengl, cpp, cmake
.. category:
.. link:
.. description: Save OpenGL frame to image
.. type: text
-->

I want to create GIF from for OpenGL Application.

I can split this problem into two steps:
1. Capture OpenGL frames.
2. Convert captured frames to GIF.

#### Capture OpenGL frames:
- The first idea comes to me is to take a sequence of snapshots manually.
`Hmm... I won't capture every frame and It needs a lot of work and sync too.`
- Second idea is to capture every frame using OpenGL itself. `yaaay`

OpenGL render to default framebuffer which is created with context to display it into the screen.
So we can use framebuffer functions to read from it.
`glReadBuffer` is a function used to read pixels from framebuffers.
We can use it with default framebuffer too.

> "Talking is cheap show me code." Linus Torvalds

```cpp
// constexpr auto g_uWidth    = 800U;
// constexpr auto g_uHeight   = 600U;
// constexpr auto g_uChannels = 3U;

auto running = true;

auto index = 0;

std::vector<std::uint8_t> frame(g_uWidth*g_uHeight*g_uChannels, 0);

while(running) {
  processInput();
  update();
  draw();
  swapBuffer();

  auto pData = static_cast<void*>(frame.data());
  // Read rectange of pixels:
  // starting point (0, 0)
  // rectange dimention (width, height)
  // Pixel format RGB
  // Pixel type unsigned byte == std::uint8_t
  glReadPixels(0, 0, g_uWidth, g_uHeight, GL_RGB, GL_UNSIGNED_BYTE, pData);

  // save frame to hard. I am using STB library
  char imageName[256];
  std::snprintf(imageName, 256, "Screenshot%02d.png", index++);
  stbi_write_png(imageName, g_uWidth, g_uHeight, g_uChannels, pData, g_uWidth * g_uChannels);
}
```

#### Convert frames to GIF:

I will use ImageMagick to convert saved to frames to GIF.
```bash
convert -loop 0 *.png screenshot.gif
```

#### Refernce:
- [Default Framebuffer](https://www.khronos.org/opengl/wiki/Default_Framebuffer)
- [ImageMagick GIF](https://imagemagick.org/script/animate.php)

