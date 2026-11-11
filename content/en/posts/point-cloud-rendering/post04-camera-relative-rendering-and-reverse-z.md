---
title: "Post 04: Camera-Relative Rendering and Reverse-Z"
date: 2026-10-14T13:00:00+03:00
weight: 5
draft: true
toc: false
images:
series: ["point-cloud-rendering"]
tags:
  - opengl
  - cpp
  - gpu
  - point-cloud
  - precision
---

[Post 02](../post02-data-layout-for-gpu-point-clouds/) stored each cloud as a double origin plus float offsets. [Post 03](../post03-one-big-pool/) put those offsets in a pool. This post closes the precision story on the transform side: how a matrix built in double lets several clouds with different origins draw together without jitter, and how reverse-Z with a 32-bit float depth buffer keeps depth precise from a metre away to the horizon.

## The problem: float matrices at survey coordinates

A vertex shader computes `mvp * pos` in float. If `mvp` contains a translation of 5 000 000 m (a UTM northing), the translation term has a resolution of half a metre (see the table in Post 02), and subtracting it from the camera position cancels almost all the significant bits. The symptom is the classic jitter: points swim by tens of centimetres as the camera moves. It is not a driver bug and no depth-buffer trick fixes it.

The rule that fixes it: **never let a large number enter a float.** Do all arithmetic that involves absolute coordinates in double, and only hand the GPU numbers that are small *relative to the camera*.

## The camera in double

The `Camera` is pure double and knows nothing about GL:

```cpp
struct Camera {
    glm::dvec3 target{0.0};
    double yaw, pitch, distance;       // orbit around the target; the eye sits `distance` away
    float  fov;                        // vertical, radians
    bool   ortho, fly;
    std::optional<glm::dvec3> pivot;
    glm::dvec3 lo{-1.0}, hi{1.0};      // scene bounds
};
glm::dvec3 camera_eye(const Camera& cam);    // target + distance * dir(yaw, pitch)
glm::dmat4 camera_view(const Camera& cam);   // rotation only: the eye is at the origin
glm::dmat4 camera_proj(const Camera& cam, double aspect);
```

Notice what `camera_view` does: it returns a **rotation only**. The eye is not translated to the origin inside the matrix. The translation is applied per cloud, below.

"Scene space" is itself relative: a shared scene origin is chosen when the first cloud loads, and `model_offset(cloud) = cloud.origin - scene.origin`, computed in double. For adjacent Toronto tiles this is what makes them line up (ROADMAP 2.0).

## The per-cloud matrix

The whole trick is one line:

```cpp
glm::mat4 App::cloud_mvp(const SceneCloud& e, const glm::dmat4& vp, glm::dvec3 eye)
{
    return glm::mat4(glm::translate(vp, model_offset(e.cloud) - eye) * cloud_xf(e));
}
```

Read it right to left:

1. `cloud_xf(e)` is the cloud's user transform (the gizmo's move/rotate/scale).
2. `model_offset(cloud) - eye` is the cloud's origin expressed **relative to the camera eye**, subtracted in double. For a cloud near the camera this is a small number, whatever the absolute coordinates are.
3. `vp` is the rotation and the projection.
4. Only now is the result cast to `float`.

The offsets in the buffer are already small (Post 02), and the translation in the matrix is already small (this step), so the float product `mvp * vec4(pos, 1)` loses nothing meaningful. A cloud that is 3 km away has a 3 km translation, which a float holds to a fraction of a millimetre. A cloud that is 5 000 km from the scene origin but under the camera has a near-zero one.

This is a per-cloud matrix, so it costs one 4x4 upload per cloud per frame, and the cloud's `pos` buffer never changes when the camera moves. The same matrix feeds the raster, the edit passes, the selection and the pick (all derived from the same function), so they agree with each other.

## Projection with an infinite far plane

The perspective matrix is written directly in double:

```cpp
const double f = 1.0 / std::tan(cam.fov * 0.5);
proj[0][0] = f / aspect;
proj[1][1] = f;
proj[3][2] = camera_near(cam);   // z_clip = near
proj[2][3] = -1.0;               // w_clip = -z_eye = view distance
```

So `z_clip = near` for every point and `w_clip = distance along the view axis`. After the divide, `z_ndc = near / distance`: **1 at the near plane, 0 at infinity**. There is no far plane to choose, and nothing is ever far-clipped. The near plane scales with the zoom, `max(distance * 0.001, 1e-3)`, so close-ups don't clip into the surface and far-away views keep their depth resolution.

Why infinite? A point cloud has no sensible far plane: the user can fit a whole city or fly to a wall. Picking the far plane as `diameter * 2` just moves the problem, and an infinite one removes a tunable that can go wrong.

## Reverse-Z in four settings

Depth that goes 1 → 0 with distance is *reverse-Z*. It takes four settings to get right, and the editor sets them in one place:

```cpp
glClipControl(GL_LOWER_LEFT, GL_ZERO_TO_ONE);            // init
glBindFramebuffer(GL_FRAMEBUFFER, target.fbo);           // attachment: GL_DEPTH_COMPONENT32F
glDepthFunc(GL_GREATER);                                 // every frame
const float depth0 = 0.0f;
glClearNamedFramebufferfv(target.fbo, GL_DEPTH, 0, &depth0);   // far = 0
```

- **`GL_ZERO_TO_ONE`** keeps clip-space z in [0, 1] instead of OpenGL's default [-1, 1]. The default range is remapped to [0, 1] in the viewport transform *after* the float rounding, which throws away the precision you wanted from reverse-Z. This is the one that is usually forgotten.
- **A 32-bit float depth attachment** (`GL_DEPTH_COMPONENT32F`). The whole benefit comes from floats being dense near 0.
- **`GL_GREATER`** and **clear to 0**, because nearer is now larger and far is 0.
- **All other GL state is set every frame.** ImGui does not save the depth function, depth mask, clip control or FBO binding, so the scene pass re-sets them.

### Why this is more precise

The depth `near / d` is hyperbolic in distance `d`: most of the [0, 1] range is spent close to the camera. With the conventional mapping (near = 0, far = 1) the float format is also densest near 0, so both effects pile up on the near plane and the far field gets coarse steps. Reverse-Z puts the *far* end at 0, where floats are densest, so the two non-linearities cancel and precision is nearly uniform in log(distance). The practical result: with an infinite far plane and a 24-bit-mantissa float, depth fighting disappears from a 1 m close-up to a 10 km horizon, in the same frame.

(I am quoting this as a well-known property of reverse-Z with float depth, not as an editor-specific measurement. The editor's own check is that the compute path and `GL_POINTS` produce the same depth buffer: bit-identical except for 3 577 of 3.24 M pixels at fit-all, about 0.1 %. Those are the same pixels whose colour differs, depth ties and one-pixel coverage shifts, which [Post 11](../post11-image-parity/) takes apart; ROADMAP 3.2.)

## Why the compute rasterizer cares

The compute rasterizer in Post 9 never touches the hardware depth buffer for points. It packs the **view distance** `w` into the key:

```glsl
const vec4 c = mvp * vec4(pos[j], pos[j + 1], pos[j + 2], 1.0);
// perspective: w = positive view distance. smaller = closer
const uint64_t v = packUint2x32(uvec2(base + i, floatBitsToUint(c.w - orthoK * c.z)));
if (v < fb[p]) atomicMin(fb[p], v);
```

Non-negative floats sort like their integer bits, so `atomicMin` on `floatBits(w) << 32 | index` is a depth test with an index tie-break. The resolve pass then converts back with the same reverse-Z formula and writes a normal depth value:

```glsl
gl_FragDepth = ortho ? 1.0 - uintBitsToFloat(v.y) : near / uintBitsToFloat(v.y);
```

That is the payoff of this post: because the resolve writes reverse-Z depth into the **same FBO** with `GL_GREATER`, everything that is still drawn with hardware depth testing (the grid, wire boxes, measure markers, the gizmo) composes correctly with the software-rasterized points, in either renderer. The fit-all view needs no special case.

## Orthographic, in the same convention

The orthographic projection follows the same rules so the rest of the code does not branch: depth range `[near, far]` along the view is mapped so near gives `z = 1` and far gives `z = 0`, with `w = 1`. The range is the scene bounds plus 10 times the larger of the distance and the scene radius around the target (so neither the grid nor anything near the camera is clipped), and the sort key becomes `1 - z`. The half height is `distance * tan(fov / 2)`, the same as the perspective view at the target, so toggling the mode keeps the size at the target.

## Checklist for your renderer

1. Keep the camera, the cloud origins and the model offsets in **double**.
2. Build `mvp = vp * translate(cloud_origin - eye) * user_xf` in double. Cast to float last. Make the view rotation-only.
3. Use an **infinite-far** projection with the near plane tied to zoom.
4. Set `glClipControl(.., GL_ZERO_TO_ONE)`, a **`D32F`** depth attachment, **`GL_GREATER`** and **clear to 0**.
5. Re-set GL state every frame if a UI library shares the context.
6. If you rasterize in compute, sort on view distance, and write reverse-Z depth in the resolve so hardware-drawn overlays still occlude correctly.

Next: even with perfect transforms, the order points sit in memory decides how much of that 23 ms you can recover: [Post 05](../post05-morton-order/).

## Where this lives in the code

`src/core/camera.hpp` and `camera.cpp` (the double camera, `camera_proj`, `camera_near`, `camera_depth_range`), `src/app/scene/render_scene.cpp` (`cloud_mvp`, per-frame GL state), `src/app/init.cpp` (`glClipControl`), `src/render/render.cpp` (raster key and resolve depth). Context: ROADMAP 2.0, R3 (M2a) and the orthographic notes in V; `render_model` for the CPU references.
