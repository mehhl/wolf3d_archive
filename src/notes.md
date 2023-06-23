Okay, so how are the assets for video generation collected?

Video generation happens in a call to `evaluate()`. This call passes a `logger.video` object. I suspect it comes from this property of the Logger object:

```{python}
class Logger(object):
# ...
        @property
        def video(self):
                return self._video
```

and this `self._video` is a `VideoRecorder` object, also from `logger.py`. But `VideoRecorder` is apparently just a wrapper to `env.render`, called via `video.record(env)`, adding some utility options like fps and render size. (It's good that they add those -- I won't have to think about how to adjust them myself!)

Okay, so how does `env.render` work? First let's check if it's overwritten by any of our env wrappers. We check this in `env.py`. 

And it is! In `TimeStepToGymWrapper`, you get this:

```{python}
 def render(self, mode='rgb_array', width=384, height=384, camera_id=0):
  camera_id = dict(quadruped=2).get(self.domain, camera_id)
  return self.env.physics.render(height, width, camera_id)
```

Well I don't like this one bit. Evidently, the program makes cameras inflexible; a single call to `video.record(env)` will generate only a single video, because there's only one camera id getting generated in the wrapper. And apparently it's a fixed one taken from `dm_control`'s dictionary, which I'm not setting up for my task. (That still doesn't explain the missing assets.)

So now we also know that `env.render()` is just a wrapper for `env.physics.render()`, and so we need to look at how it's implemented in physics. I'll probably have to:
1. Implement a `render()` method in my Physics in `hide.py`;
2. Make sure that the correct cameras are called from `env.render()` in `env.py`;
3. Make sure that the assets I've so painstakingly loaded in `get_model_and_agents()` in `hide.py`
are actually being loaded.

So the `render()` method is actually in `dm_control/mujoco/engine.py`. The camera is identified from the model through camera id. Here's the important part:

```{python}
    camera = Camera(
        physics=self,
        height=height,
        width=width,
        camera_id=camera_id,
        scene_callback=scene_callback)
    image = camera.render(
        overlays=overlays, depth=depth, segmentation=segmentation,
        scene_option=scene_option, render_flag_overrides=render_flag_overrides)
    camera._scene.free()  # pylint: disable=protected-access
    return image
```

The `Camera` here is probably a Python wrapper for the camera ("rendering context" lol) that already exists on the C level, and I probably shouldn't touch it. Instead, I should correct the camera id to get input from two cameras, and I should correct the cameras within `hide.get_model_and_agents()`. o

I've already specified `depth`, etc in the model, so it would be pretty annoying if it turned out that was being overridden because of bad design.

Okay. So what is the camera id being passed? Apparently it's 0, corresponding to `learner/head`. o

Alright, so it turns out my initial solution was just incorrect. Not surprising! What I can do in order to simulate a PoV effect is to create a site that will correspond to my intended camera position, then orient that site in accordance with dog's head, and then attach a camera to that site whose orientation will correspond to the site's (and hence the head's) orientation. I should first test this out in raw XML, so I don't waste time recompiling `hide.py` all the time.

Okay, regarding the camera, I now have a definite path towards success. I have set up a camera `PoV_somtething` and I need to adjust its Euler angle to have it display what I want. This is a simple iterative process that will take me a while, but is sure to work eventually. I also know where to adjust the camera parameters now; when I come to Cracow, I'll write a wrapper for this. I'll probably have to adjust the seekers' light orientation because it is sure not to work correctly for now.

Another important thing I need to figure out at least a preliminary solution for is the assets. Why the hell are they loading from a different directory? This might be more difficult to work out. :(

Going forward:

1. Figure out how to make the simulation load correct assets. (Floor etc) For now it looks like its defaulting.
Maybe just hardcode that shit lol. Honestly the most reasonable solution, if it weren't for the fact that you can't do this 
because you have to create the original model in PyMJCF.
2. Figure out the right camera angles and rendering parameters. You want a far view, so that some seekers are visible in the distance.
3. Figure out the texture! You need to have it loaded properly.

4. Train the model as it is currently.

Okay, resolution fixed. I now can generate video in glorious 1920x1080p. Now we need to fix frame rate, I want 60 fps.

