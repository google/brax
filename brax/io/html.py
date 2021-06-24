# Copyright 2021 The Brax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Exports a system config and trajectory as an html view."""

import json
from typing import List

import brax
from brax.io.json import JaxEncoder
from tensorflow.io import gfile


from google.protobuf import json_format


def save_html(path: str, sys: brax.System, qps: List[brax.QP]):
  """Saves trajectory as a HTML file."""
  with gfile.GFile(path, 'wb') as fout:
    fout.write(render(sys, qps))


def render(sys: brax.System, qps: List[brax.QP]) -> str:
  if any((len(qp.pos.shape), len(qp.rot.shape)) != (2, 2) for qp in qps):
    raise RuntimeError('unexpected shape in qp.')
  d = {
      'config': json_format.MessageToDict(sys.config, True),
      'pos': [qp.pos for qp in qps],
      'rot': [qp.rot for qp in qps],
  }
  system = json.dumps(d, cls=JaxEncoder)
  return _HTML.replace('<!-- system json goes here -->', system)


_HTML = """
<html>

  <head>
    <title>Brax visualizer</title>
    <style>
body { margin: 0; }

#threejs-view >div { width: 700px; height: 428px;}

#media-player > div{min-height: 28px; height: 28px; width: 700px; }
#media-player > div > ul  {position: absolute; z-index: 0;}
#media-player > div > ul { display: table-row;  }
#media-player > div > ul > li { display: table-cell; min-width: 40px ;padding-left: 0px;padding-right:1px}
#media-player > div > ul > li.cr { background: none; line-height: normal; vertical-align: top; border-left: 0px !important;border-bottom: 0px}
#media-player > div > ul > li.cr >div { background: #1a1a1a; height: 27px !important;padding-left: 6px;}
#media-player > div > ul > li.cr > div > span.property-name{padding-top: 6px; }
#media-player > div > ul > li.cr.string >div { border-left:4px solid green; }
#media-player > div > ul > li.cr.number { width: 660px; }
#media-player > div > ul > li.cr.number .c { width: 100%; }
#media-player > div > ul > li.cr.number .slider { width: 100%; }
#media-player > div > ul > li.cr.number .property-name { display: none; }
#media-player > div > ul > li.cr.number input { display: none; }

#media-player > div > ul > li.cr.function >div { border-left:4px solid red;}
#media-player > div > ul > li.cr.color >div { border-left:4px solid orange;}
#media-player > div > ul > li.cr.boolean>div { border-left:4px solid purple;}
#media-player .close-button { display:none;}

.dg .close-button { display:none;}
.dg.a { margin-left:455px !important; float: left; }
    </style>
  </head>

  <script type="application/javascript">
  var system = <!-- system json goes here -->;
  </script>

  <body>

  <div id="overall">
    <div id="threejs-view"></div>
    <div id="media-player"></div>
  </div>
    <script type="module">
import * as THREE from 'https://threejs.org/build/three.module.js';

const basicMaterial = new THREE.MeshPhongMaterial({color: 0x665544});
const targetMaterial = new THREE.MeshPhongMaterial({color: 0xff2222});
const hoverMaterial = new THREE.MeshPhongMaterial({color: 0x332722, emissive: 0x114a67});
const selectMaterial = new THREE.MeshPhongMaterial({color: 0x2194ce});

function setHover(group, hovering) {
  group.traverse(function(child) {
    if (child instanceof THREE.Mesh) {
      child.material = group.selected ? selectMaterial : hovering ? hoverMaterial : basicMaterial;
    }
  });
}

function setSelect(group, select) {
  group.selected = select;
  console.log(new THREE.Euler().setFromQuaternion( group.getWorldQuaternion()));
  group.traverse(function(child) {
    if (child instanceof THREE.Mesh) {
      child.material = select ? selectMaterial : basicMaterial;
    }
  });
}

function createCapsule(capsule) {
  const sphere_geom = new THREE.SphereGeometry(capsule.radius, 16, 16);
  const cylinder_geom = new THREE.CylinderGeometry(
      capsule.radius, capsule.radius, capsule.length - 2 * capsule.radius);

  const sphere1 = new THREE.Mesh(sphere_geom, basicMaterial);
  sphere1.position.set(0, capsule.length / 2 - capsule.radius, 0);
  sphere1.castShadow = true;

  const sphere2 = new THREE.Mesh(sphere_geom, basicMaterial);
  sphere2.position.set(0, -capsule.length / 2 + capsule.radius, 0);
  sphere2.castShadow = true;

  const cylinder = new THREE.Mesh(cylinder_geom, basicMaterial);
  cylinder.castShadow = true;

  const group = new THREE.Group();
  group.add(sphere1, sphere2, cylinder);
  return group;
}

function createBox(box) {
  const geom = new THREE.BoxBufferGeometry(
      2 * box.halfsize.x, 2 * box.halfsize.z, 2 * box.halfsize.y);
  const mesh = new THREE.Mesh(geom, basicMaterial);
  mesh.castShadow = true;
  return mesh;
}

function createPlane(plane) {
  const group = new THREE.Group();
  const mesh = new THREE.Mesh(
      new THREE.PlaneGeometry(2000, 2000),
      new THREE.MeshPhongMaterial({color: 0x999999, depthWrite: false}));
  mesh.rotation.x = -Math.PI / 2;
  mesh.receiveShadow = true;
  group.add(mesh);

  const mesh2 = new THREE.GridHelper(2000, 2000, 0x000000, 0x000000);
  mesh2.material.opacity = 0.4;
  mesh2.material.transparent = true;
  group.add(mesh2);
  return group;
}

function createSphere(sphere, name) {
  const geom = new THREE.SphereGeometry(sphere.radius, 16, 16);
  let mat = name.toLowerCase() == 'target' ? targetMaterial : basicMaterial;
  const mesh = new THREE.Mesh(geom, mat);
  mesh.castShadow = true;
  return mesh;
}

function createScene(system) {
  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0xa0a0a0);
  scene.fog = new THREE.Fog(0xa0a0a0, 10, 50);

  const hemiLight = new THREE.HemisphereLight(0xffffff, 0x444444);
  hemiLight.position.set(0, 20, 0);
  scene.add(hemiLight);

  const dirLight = new THREE.DirectionalLight(0xffffff);
  dirLight.position.set(3, 10, 10);
  dirLight.castShadow = true;
  dirLight.shadow.camera.top = 10;
  dirLight.shadow.camera.bottom = -10;
  dirLight.shadow.camera.left = -10;
  dirLight.shadow.camera.right = 10;
  dirLight.shadow.camera.near = 0.1;
  dirLight.shadow.camera.far = 40;
  scene.add(dirLight);

  system.config.bodies.forEach(function(body) {
    const parent = new THREE.Group();
    parent.name = body.name.replaceAll('/', '_'); // sanitize node name
    body.colliders.forEach(function(collider) {
      let child;
      if ('box' in collider) {
        child = createBox(collider.box);
      } else if ('capsule' in collider) {
        child = createCapsule(collider.capsule);
      } else if ('plane' in collider) {
        child = createPlane(collider.plane);
      } else if ('sphere' in collider) {
        child = createSphere(collider.sphere, body.name);
      }
      // note the minus signs--these are to convert from brax's z-up
      // coordinate system into threejs's y-up coordinates
      if (collider.rotation) {
        // convert from z-up to y-up coordinate system
        const rot = new THREE.Vector3(collider.rotation.x, collider.rotation.y,
                                      collider.rotation.z);
        rot.multiplyScalar(Math.PI / 180);
        const eul = new THREE.Euler();
        eul.setFromVector3(rot);
        child.quaternion.setFromEuler(eul);
        child.quaternion.x = -child.quaternion.x;
        const tmp = child.quaternion.y;
        child.quaternion.y = -child.quaternion.z;
        child.quaternion.z = -tmp;
      }
      if (collider.position) {
        child.position.set(collider.position.x, collider.position.z,
                           collider.position.y);
      }
      parent.add(child);
    });
    scene.add(parent);
  });

  return scene;
}

function createAnimationClip(system) {
  const times = [...Array(system.pos.length).keys()].map(x => x * system.config.dt);
  const tracks = [];

  system.config.bodies.forEach(function(body, bi) {
    const group = body.name.replaceAll('/', '_'); // sanitize node name
    const pos = system.pos.map(p => [p[bi][0], p[bi][2], p[bi][1]]);
    const rot =
        system.rot.map(r => [-r[bi][1], -r[bi][3], -r[bi][2], r[bi][0]]);
    tracks.push(new THREE.VectorKeyframeTrack(
        'scene/' + group + '.position', times, pos.flat()));
    tracks.push(new THREE.QuaternionKeyframeTrack(
        'scene/' + group + '.quaternion', times, rot.flat()));
  });

  return new THREE.AnimationClip('Action', -1, tracks);
}

import {EventDispatcher, Raycaster, Vector2} from 'https://threejs.org/build/three.module.js';

const Selector = function(_objects, _camera, _domElement) {
  const _raycaster = new Raycaster();
  const _mouse = new Vector2();
  const _intersections = [];
  let _selected = null, _hovered = null;
  let _drag = false;
  const scope = this;

  function activate() {
    _domElement.addEventListener('pointermove', onPointerMove);
    _domElement.addEventListener('pointerdown', onPointerDown);
    _domElement.addEventListener('pointerup', onPointerUp);
  }

  function deactivate() {
    _domElement.removeEventListener('pointermove', onPointerMove);
    _domElement.removeEventListener('pointerdown', onPointerDown);
    _domElement.removeEventListener('pointerup', onPointerUp);
    _domElement.style.cursor = '';
  }

  function dispose() {
    deactivate();
  }

  function getObjects() {
    return _objects;
  }

  function onPointerMove(event) {
    event.preventDefault();
    _drag = true;
    const rect = _domElement.getBoundingClientRect();
    _mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
    _mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
    _intersections.length = 0;
    _raycaster.setFromCamera(_mouse, _camera);
    _raycaster.intersectObjects(_objects, true, _intersections);

    if (_intersections.length > 0) {
      let object = _intersections[0].object;
      while (object.parent && !object.name) {
        object = object.parent;
      }
      if (_hovered !== object) {
        if (_hovered) {
          scope.dispatchEvent({type: 'hoveroff', object: _hovered});
        }
        _hovered = object;
        scope.dispatchEvent({type: 'hoveron', object: _hovered});
        _domElement.style.cursor = 'pointer';
      }
    } else if (_hovered !== null) {
      scope.dispatchEvent({type: 'hoveroff', object: _hovered});

      _domElement.style.cursor = 'auto';
      _hovered = null;
    }
  }

  function onPointerDown(event) {
    event.preventDefault();
    _drag = false;
  }

  function onPointerUp(event) {
    event.preventDefault();
    if (_drag) return; // ignore drag events, only handle clicks
    _intersections.length = 0;
    _raycaster.setFromCamera(_mouse, _camera);
    _raycaster.intersectObjects(_objects, true, _intersections);

    if (_intersections.length > 0) {
      let object = _intersections[0].object;
      while (object.parent && !object.name) {
        object = object.parent;
      }
      if (_selected !== object) {
        if (_selected) {
          scope.dispatchEvent({type: 'deselect', object: _selected});
        }
        _selected = object;
        scope.dispatchEvent({type: 'select', object: _selected});
      }
    } else if (_selected !== null) {
      scope.dispatchEvent({type: 'deselect', object: _selected});
      _selected = null;
    }
  }

  activate();

  // API
  this.enabled = true;
  this.activate = activate;
  this.deactivate = deactivate;
  this.dispose = dispose;
  this.getObjects = getObjects;
};

Selector.prototype = Object.create(EventDispatcher.prototype);
Selector.prototype.constructor = Selector;

import {OrbitControls} from 'https://threejs.org/examples/jsm/controls/OrbitControls.js';
import {GUI} from 'https://threejs.org/examples/jsm/libs/dat.gui.module.js';

// create scene and trajectory animation
const clock = new THREE.Clock();
const scene = createScene(system);
const clip = createAnimationClip(system);
const mixer = new THREE.AnimationMixer(scene);
const action = mixer.clipAction(clip);
action.play();

// build camera and renderer
const camera = new THREE.PerspectiveCamera(40, 700 / 500, 1, 100);
camera.position.set(5, 2, 8);

const renderer = new THREE.WebGLRenderer();
const mediaPlayerMargin = clip.duration > 0 ? 28 : 0;
renderer.setPixelRatio(window.devicePixelRatio);
renderer.setSize(700, 500);
renderer.outputEncoding = THREE.sRGBEncoding;
renderer.shadowMap.enabled = true;
document.getElementById('threejs-view').appendChild(renderer.domElement);

window.onresize = function() {
  const width = 700;
  const height = 500;

  camera.aspect = width / height;
  camera.updateProjectionMatrix();
  renderer.setSize(width, height);
};

// build up gui
const orbiter = new OrbitControls(camera, renderer.domElement);
if (clip.duration == 0) {
  orbiter.autoRotate = true;
  orbiter.autoRotateSpeed = 1;
}
orbiter.enablePan = false;
orbiter.enableDamping = true;

const ignore = ['target', 'ground', 'floor'];
const selectable = scene.children.filter(
    o => o instanceof THREE.Group && !ignore.includes(o.name.toLowerCase()));
const center = selectable[0];
const selector = new Selector(selectable, camera, renderer.domElement);
let selected = null;

selector.addEventListener('hoveron', function(event) {
  setHover(event.object, true);
});

selector.addEventListener('hoveroff', function(event) {
  setHover(event.object, false);
});

selector.addEventListener('select', function(event) {
  setSelect(event.object, true);
  selected = event.object;
  bodyInspector.show();
});

selector.addEventListener('deselect', function(event) {
  setSelect(event.object, false);
  bodyInspector.hide();
  selected = null;
});

const bodyInspector = new GUI();
const bodySettings = {
  'name': '',
  'pos.x': 0.0,
  'pos.y': 0.0,
  'pos.z': 0.0,
  'rot.w': 0.0,
  'rot.x': 0.0,
  'rot.y': 0.0,
  'rot.z': 0.0,
};
const controllers = [];
controllers.push(bodyInspector.add(bodySettings, 'name'));
const posFolder = bodyInspector.addFolder('Position');
posFolder.open();
controllers.push(posFolder.add(bodySettings, 'pos.x').name('x').step(0.01));
controllers.push(posFolder.add(bodySettings, 'pos.y').name('y').step(0.01));
controllers.push(posFolder.add(bodySettings, 'pos.z').name('z').step(0.01));
const rotFolder = bodyInspector.addFolder('Rotation');
rotFolder.open();
controllers.push(rotFolder.add(bodySettings, 'rot.w').name('w').step(0.01));
controllers.push(rotFolder.add(bodySettings, 'rot.x').name('x').step(0.01));
controllers.push(rotFolder.add(bodySettings, 'rot.y').name('y').step(0.01));
controllers.push(rotFolder.add(bodySettings, 'rot.z').name('z').step(0.01));
controllers.forEach(o => o.domElement.style.pointerEvents = 'none');
controllers.forEach(o => o.listen());
bodyInspector.hide();

if (clip.duration > 0) {
  const mediaPlayer = new GUI({autoPlace: false});
  const mediaActions = {
    playPause: function() {
      action.paused = !action.paused;
      playPause.name(action.paused ? '▶' : '| |');
    },
  };
  const playPause = mediaPlayer.add(mediaActions, 'playPause');
  playPause.name('| |');
  mediaPlayer.add(action, 'time', 0, clip.duration, 0.01).listen();
  document.getElementById('media-player').appendChild(mediaPlayer.domElement);
}

function animate() {
  requestAnimationFrame(animate);
  const delta = clock.getDelta();
  mixer.update(delta);

  // gently (with damping) move the orbiter target to the selected or center
  const newTarget = new THREE.Vector3();
  (selected ?? center).getWorldPosition(newTarget);
  orbiter.target.x += (newTarget.x - orbiter.target.x) / 10;
  orbiter.target.y = 0;
  orbiter.target.z += (newTarget.z - orbiter.target.z) / 10;
  orbiter.update();
  renderer.render(scene, camera);

  // if the target gets too far from the camera, move the camera
  if (camera.position.distanceTo(newTarget) > 10) {
    camera.position.add(newTarget.sub(camera.position).multiplyScalar(0.01));
  }

  if (selected !== null) {
    bodySettings['name'] = selected.name;
    const pos = new THREE.Vector3();
    selected.getWorldPosition(pos);
    bodySettings['pos.x'] = pos.x;
    bodySettings['pos.y'] = pos.y;
    bodySettings['pos.z'] = pos.z;
    const rot = new THREE.Quaternion();
    selected.getWorldQuaternion(rot);
    bodySettings['rot.w'] = rot.w;
    bodySettings['rot.x'] = rot.x;
    bodySettings['rot.y'] = rot.y;
    bodySettings['rot.z'] = rot.z;
  }
}

animate();

    </script>
  </body>
</html>
"""
