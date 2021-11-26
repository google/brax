/**
 * @fileoverview BraxViewer can render static trajectories from json and also
 * connect to a remote brax engine for interactive visualization.
 */

import * as THREE from 'https://threejs.org/build/three.module.js';
import {OrbitControls} from 'https://threejs.org/examples/jsm/controls/OrbitControls.js';
import {GUI} from 'https://threejs.org/examples/jsm/libs/lil-gui.module.min.js';

import {Animator} from './animator.js';
import {Selector} from './selector.js';
import {createScene, createTrajectory} from './system.js';

function downloadDataUri(name, uri) {
  let link = document.createElement('a');
  link.download = name;
  link.href = uri;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
}

function downloadFile(name, contents, mime) {
  mime = mime || 'text/plain';
  let blob = new Blob([contents], {type: mime});
  let link = document.createElement('a');
  document.body.appendChild(link);
  link.download = name;
  link.href = window.URL.createObjectURL(blob);
  link.onclick = function(e) {
    let scope = this;
    setTimeout(function() {
      window.URL.revokeObjectURL(scope.href);
    }, 1500);
  };
  link.click();
  link.remove();
}

const hoverMaterial =
    new THREE.MeshPhongMaterial({color: 0x332722, emissive: 0x114a67});
const selectMaterial = new THREE.MeshPhongMaterial({color: 0x2194ce});

class Viewer {
  constructor(domElement, system) {
    this.domElement = domElement;
    this.system = system;
    this.scene = createScene(system);
    this.trajectory = createTrajectory(system);

    /* set up renderer, camera, and add default scene elements */
    this.renderer = new THREE.WebGLRenderer({antialias: true, alpha: true});
    this.renderer.setPixelRatio(window.devicePixelRatio);
    this.renderer.shadowMap.enabled = true;
    this.renderer.outputEncoding = THREE.sRGBEncoding;

    this.domElement.appendChild(this.renderer.domElement);

    this.camera = new THREE.PerspectiveCamera(40, 1, 0.01, 100);
    this.camera.position.set(5, 2, 8);
    this.camera.follow = true;
    this.camera.freezeAngle = false;
    this.camera.followDistance = 10;

    this.scene.background = new THREE.Color(0xa0a0a0);
    this.scene.fog = new THREE.Fog(0xa0a0a0, 40, 60);

    const hemiLight = new THREE.HemisphereLight(0xffffff, 0x444444);
    hemiLight.position.set(0, 20, 0);
    this.scene.add(hemiLight);

    const dirLight = new THREE.DirectionalLight(0xffffff);
    dirLight.position.set(3, 10, 10);
    dirLight.castShadow = true;
    dirLight.shadow.camera.top = 10;
    dirLight.shadow.camera.bottom = -10;
    dirLight.shadow.camera.left = -10;
    dirLight.shadow.camera.right = 10;
    dirLight.shadow.camera.near = 0.1;
    dirLight.shadow.camera.far = 40;
    this.scene.add(dirLight);

    /* set up orbit controls */
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enablePan = false;
    this.controls.enableDamping = true;
    this.controls.addEventListener('start', () => {this.setDirty();});
    this.controls.addEventListener('change', () => {this.setDirty();});
    this.controlTargetPos = this.controls.target.clone();

    /* set up gui */
    this.gui = new GUI({autoPlace: false});
    this.domElement.parentElement.appendChild(this.gui.domElement);
    this.gui.domElement.style.position = 'absolute';
    this.gui.domElement.style.right = 0;
    this.gui.domElement.style.top = 0;

    /* add camera inspectors */
    const cameraFolder = this.gui.addFolder('Camera');
    cameraFolder.add(this.camera, 'freezeAngle').name('Freeze Angle');
    cameraFolder.add(this.camera, 'follow').name('Follow Target');
    cameraFolder.add(this.camera, 'followDistance')
        .name('Follow Distance')
        .min(1)
        .max(50);

    /* set up animator and load trajectory */
    this.animator = new Animator(this);
    this.animator.load(this.trajectory, {});

    /* add body insepctors */
    const bodiesFolder = this.gui.addFolder('Bodies');
    bodiesFolder.open();

    this.bodyFolders = {};

    for (let c of this.scene.children) {
      if (!c.name) continue;
      const folder = bodiesFolder.addFolder(c.name);
      this.bodyFolders[c.name] = folder;

      function defaults() {
        for (const gui of arguments) {
          gui.step(0.01).listen().domElement.style.pointerEvents = 'none';
        }
      }
      defaults(
          folder.add(c.position, 'x').name('pos.x'),
          folder.add(c.position, 'y').name('pos.y'),
          folder.add(c.position, 'z').name('pos.z'),
          folder.add(c.rotation, 'x').name('rot.x'),
          folder.add(c.rotation, 'y').name('rot.y'),
          folder.add(c.rotation, 'z').name('rot.z'),
      );
    }
    let saveFolder = this.gui.addFolder('Save / Capture');
    saveFolder.add(this, 'saveScene').name('Save Scene');
    saveFolder.add(this, 'saveImage').name('Capture Image');

    this.gui.close();

    /* set up body selector */
    this.selector = new Selector(this);
    this.selector.addEventListener(
        'hoveron', (evt) => this.setHover(evt.object, true));
    this.selector.addEventListener(
        'hoveroff', (evt) => this.setHover(evt.object, false));
    this.selector.addEventListener(
        'select', (evt) => this.setSelected(evt.object, true));
    this.selector.addEventListener(
        'deselect', (evt) => this.setSelected(evt.object, false));

    this.defaultTarget = this.selector.selectable[0];
    this.target = this.defaultTarget;

    /* get ready to render first frame */
    this.setDirty();

    window.onload = (evt) => this.setSize();
    window.addEventListener('resize', (evt) => this.setSize(), false);
    requestAnimationFrame(() => this.setSize());

    const resizeObserver = new ResizeObserver(() => this.resizeCanvasToDisplaySize());
    resizeObserver.observe(this.domElement, {box: 'content-box'});

    /* start animation */
    this.animate();
  }

  setDirty() {
    this.needsRender = true;
  }

  setSize(w, h) {
    if (w === undefined) {
      w = this.domElement.offsetWidth;
    }
    if (h === undefined) {
      h = window.innerHeight;
    }
    if (this.camera.type == 'OrthographicCamera') {
      this.camera.right =
          this.camera.left + w * (this.camera.top - this.camera.bottom) / h;
    } else {
      this.camera.aspect = w / h;
    }
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(w, h);
    this.setDirty();
  }

  resizeCanvasToDisplaySize() {
    //look up canvas size
    const width = this.domElement.clientWidth;
    const height = this.domElement.clientHeight;
    this.setSize(width, height);
  }

  render() {
    this.renderer.render(this.scene, this.camera);
    this.needsRender = false;
  }

  animate() {
    requestAnimationFrame(() => this.animate());
    this.animator.update();

    // make sure the orbiter is pointed at the right target
    const targetPos = new THREE.Vector3();
    this.target.getWorldPosition(targetPos);

    // if the target gets too far from the camera, nudge the camera
    if (this.camera.follow) {
      this.controls.target.lerp(targetPos, 0.1);
      if (this.camera.position.distanceTo(this.controls.target) >
          this.camera.followDistance) {
        const followBehind = this.controls.target.clone()
                                 .sub(this.camera.position)
                                 .normalize()
                                 .multiplyScalar(this.camera.followDistance)
                                 .sub(this.controls.target)
                                 .negate();
        this.camera.position.lerp(followBehind, 0.5);
        this.setDirty();
      }
    }

    if (this.controls.update()) {
      this.setDirty();
    }

    // if freezeAngle requested, move the camera on xz plane to match target
    if (this.camera.freezeAngle) {
      const off = new THREE.Vector3();
      off.add(this.controls.target).sub(this.controlTargetPos);
      off.setComponent(1, 0);
      if (off.lengthSq() > 0) {
        this.camera.position.add(off);
        this.setDirty();
      }
    }
    this.controlTargetPos.copy(this.controls.target);


    if (this.needsRender) {
      this.render();
    }
  }

  saveImage() {
    this.render();
    const imageData = this.renderer.domElement.toDataURL();
    downloadDataUri('brax.png', imageData);
  }

  saveScene() {
    downloadFile('system.json', JSON.stringify(this.system));
  }

  setHover(object, hovering) {
    this.setDirty();
    if (!object.selected) {
      object.traverse(function(child) {
        if (child instanceof THREE.Mesh) {
          child.material = hovering ? hoverMaterial : child.baseMaterial;
        }
      });
    }
    const titleElement =
        this.bodyFolders[object.name].domElement.querySelector('.title');
    if (titleElement) {
      titleElement.style.backgroundColor = hovering ? '#2fa1d6' : '#000';
    }
  }

  setSelected(object, selected) {
    object.selected = selected;
    this.target = selected ? object : this.defaultTarget;
    object.traverse((child) => {
      if (child instanceof THREE.Mesh) {
        child.material = selected ? selectMaterial : child.baseMaterial;
      }
    });
    if (object.selected) {
      this.bodyFolders[object.name].open();
    } else {
      this.bodyFolders[object.name].close();
    }
    this.setDirty();
  }
}

export {Viewer};