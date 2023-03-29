import * as THREE from 'https://cdn.jsdelivr.net/gh/mrdoob/three.js@r135/build/three.module.js';

class Animator {

  constructor(viewer) {
    this.viewer = viewer;
    this.mixer = new THREE.AnimationMixer(this.viewer.scene);
    this.clock = new THREE.Clock();
    this.playing = false;
    this.time = 0;
    this.timeScrubber = null;
    this.duration = 0;
    this.loop = true;
  }

  play() {
    this.clock.start();
    this.action.play();
    this.playing = true;
  }

  pause() {
    this.clock.stop();
    this.playing = false;
  }

  playPause() {
    if (this.playing) {
      this.pause();
    } else {
      this.play();
    }
  }

  displayProgress(time) {
    this.time = time;
    if (this.timeScrubber !== null) {
      this.timeScrubber.updateDisplay();
    }
  }

  seek(time) {
    this.action.time = Math.max(0, Math.min(this.action._clip.duration, time));
    this.mixer.update(0);
    this.viewer.setDirty();
  }

  setLoop(loop) {
    this.loop = loop;
    this.action.setLoop(loop ? THREE.LoopRepeat : THREE.LoopOnce, Infinity);
  }

  reset() {
    this.action.reset();
    this.displayProgress(0);
    this.mixer.update(0);
    this.viewer.setDirty();
  }

  clear() {
    this.mixer.stopAllAction();
    this.action = null;
    this.duration = 0;
    this.displayProgress(0);
    this.mixer = new THREE.AnimationMixer(this.viewer.scene);
  }

  load(trajectory, options) {
    this.clear();

    if (options.play === undefined) {
      options.play = true;
    }
    if (options.loop == undefined) {
      options.loop = true;
    }
    if (options.clampWhenFinished === undefined) {
      options.clampWhenFinished = true;
    }

    this.duration = 0;
    this.progress = 0;
    this.action = this.mixer.clipAction(trajectory);
    this.action.clampWhenFinished = options.clampWhenFinished;
    this.setLoop(options.loop);
    this.duration = trajectory.duration;

    if (this.duration > 0) {
      this.folder = this.viewer.gui.addFolder('Trajectory');
      this.folder.open();
      this.folder.add(this, 'playPause').name('Play / Pause');
      this.folder.add(this, 'reset').name('Reset');

      this.timeScrubber = this.folder.add(this, 'time', 0, this.duration, 0.001);
      this.timeScrubber.onChange((value) => this.seek(value));
      this.folder.add(this.mixer, 'timeScale').step(0.01).min(0);
      this.folder.add(this, 'loop').onChange((value) => this.setLoop(value));
    }

    this.reset();
    if (options.play) {
      this.play();
    }
  }

  update() {
    if (this.playing) {
      this.mixer.update(this.clock.getDelta());
      this.viewer.setDirty();
      if (this.duration != 0) {
        this.displayProgress(this.action.time);
      } else {
        this.displayProgress(0);
      }

      if (this.action.paused) {
        this.pause();
        this.action.reset();
      }
    }
  }

  afterRender() {
    if (this.recording) {
      this.capturer.capture(this.viewer.renderer.domElement);
    }
  }
}

export { Animator };
