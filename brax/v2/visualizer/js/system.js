import * as THREE from 'https://cdn.jsdelivr.net/gh/mrdoob/three.js@r135/build/three.module.js';


const DEBUG_OPACITY = 0.6;

function createCheckerBoard() {
  const width = 2;
  const height = 2;

  const size = width * height;
  const data = new Uint8Array(3 * size);
  const colors = [new THREE.Color(0x999999), new THREE.Color(0x888888)];

  for (let i = 0; i < size; i++) {
    const stride = i * 3;
    const ck = [0, 1, 1, 0];
    const color = colors[ck[i]];
    data[stride + 0] = Math.floor(color.r * 255);
    data[stride + 1] = Math.floor(color.g * 255);
    data[stride + 2] = Math.floor(color.b * 255);
  }
  const texture = new THREE.DataTexture(data, width, height, THREE.RGBFormat);
  texture.wrapS = THREE.RepeatWrapping;
  texture.wrapT = THREE.RepeatWrapping;
  texture.repeat.set(1000, 1000);
  return new THREE.MeshStandardMaterial({map: texture});
}

function getCapsuleAxisSize(capsule) {
  return capsule.length * 2;
}

function getSphereAxisSize(sphere) {
  return sphere.radius * 2;
}

function getBoxAxisSize(box) {
  return Math.max(box.halfsize[0], box.halfsize[1], box.halfsize[2]) * 4;
}

function getMeshAxisSize(geom) {
  let size = 1;
  for (let i = 0; i < geom.vert.length; i++) {
    let v = geom.vert[i];
    size = Math.max(v.x, v.y, v.z, size);
  }
  return size;
}

function createCapsule(capsule, mat, debug) {
  const sphere_geom = new THREE.SphereGeometry(capsule.radius, 16, 16);
  const cylinder_geom = new THREE.CylinderGeometry(
      capsule.radius, capsule.radius, capsule.length);

  const sphere1 = new THREE.Mesh(sphere_geom, mat);
  sphere1.baseMaterial = sphere1.material;
  sphere1.position.set(0, 0, capsule.length / 2);
  sphere1.castShadow = true;

  const sphere2 = new THREE.Mesh(sphere_geom, mat);
  sphere2.baseMaterial = sphere2.material;
  sphere2.position.set(0, 0, -capsule.length / 2);
  sphere2.castShadow = true;

  const cylinder = new THREE.Mesh(cylinder_geom, mat);
  cylinder.baseMaterial = cylinder.material;
  cylinder.castShadow = true;
  cylinder.rotation.x = -Math.PI / 2;

  if (debug) {
    sphere1.material.transparent = true;
    sphere2.material.transparent = true;
    cylinder.material.transparent = true;
    sphere1.material.opacity = DEBUG_OPACITY;
    sphere2.material.opacity = DEBUG_OPACITY;
    cylinder.material.opacity = DEBUG_OPACITY;
  }

  const group = new THREE.Group();
  group.add(sphere1, sphere2, cylinder);
  return group;
}

function createBox(box, mat, debug) {
  const geom = new THREE.BoxBufferGeometry(
      2 * box.halfsize[0], 2 * box.halfsize[1], 2 * box.halfsize[2]);
  const mesh = new THREE.Mesh(geom, mat);
  mesh.castShadow = true;
  mesh.baseMaterial = mesh.material;
  if (debug) {
    mesh.material.transparent = true;
    mesh.material.opacity = DEBUG_OPACITY;
  }
  return mesh;
}

function createPlane(plane, mat) {
  const geometry = new THREE.PlaneGeometry(2000, 2000);
  const mesh = new THREE.Mesh(geometry, mat);
  mesh.receiveShadow = true;
  mesh.baseMaterial = mesh.material;

  return mesh;
}

function createSphere(sphere, mat, debug) {
  const geom = new THREE.SphereGeometry(sphere.radius, 16, 16);
  const mesh = new THREE.Mesh(geom, mat);
  mesh.castShadow = true;
  mesh.baseMaterial = mesh.material;
  if (debug) {
    mesh.material.transparent = true;
    mesh.material.opacity = DEBUG_OPACITY;
  }
  return mesh;
}

function createMesh(meshGeom, mat, debug) {
  const bufferGeometry = new THREE.BufferGeometry();
  const vertices = meshGeom.vert;
  const positions = new Float32Array(vertices.length * 3);
  // Convert the coordinate system.
  vertices.forEach(function(vertice, i) {
    positions[i * 3] = vertice[0];
    positions[i * 3 + 1] = vertice[1];
    positions[i * 3 + 2] = vertice[2];
  });
  const indices = new Uint16Array(meshGeom.face.flat());
  bufferGeometry.setAttribute(
      'position', new THREE.BufferAttribute(positions, 3));
  bufferGeometry.setIndex(new THREE.BufferAttribute(indices, 1));
  bufferGeometry.computeVertexNormals();

  const mesh = new THREE.Mesh(bufferGeometry, mat);
  mesh.castShadow = true;
  mesh.baseMaterial = mesh.material;
  if (debug) {
    mesh.material.transparent = true;
    mesh.material.opacity = DEBUG_OPACITY;
  }
  return mesh;
}

function hasContactDebug(system) {
  if (system.hasOwnProperty('has_contact_debug')) {
    return system.has_contact_debug;
  }
  let maxLen = 0;
  for (let i = 0; i < system.contact_pos?.length; i++) {
    maxLen = Math.max(system.contact_pos[i].length, maxLen);
  }
  system.has_contact_debug = system.debug && (maxLen > 0);
  return system.has_contact_debug;
}

function createScene(system) {
  const scene = new THREE.Scene();
  const meshGeoms = {};
  if (system.meshes) {
    Object.entries(system.meshes).forEach(function(geom) {
      meshGeoms[geom[0]] = geom[1];
    });
  }
  if (system.debug) {
    // Add a world axis for debugging.
    const worldAxis = new THREE.AxesHelper(100);
    scene.add(worldAxis);
  }
  let minAxisSize = 1e6;
  Object.entries(system.geoms).forEach(function(geom) {
    const name = geom[0];
    const parent = new THREE.Group();
    parent.name = name.replaceAll('/', '_');  // sanitize node name
    geom[1].forEach(function(collider) {
      const color = collider.color       ? collider.color :
          name.toLowerCase() == 'target' ? '#ff2222' :
                                           '#665544';
      const mat = (collider.name == 'Plane') ?
          createCheckerBoard() :
          (collider.name == 'heightMap') ?
          new THREE.MeshStandardMaterial({color: color, flatShading: true}) :
          new THREE.MeshPhongMaterial({color: color});
      let child;
      let axisSize;
      if (collider.name == 'Box') {
        child = createBox(collider, mat, system.debug);
        axisSize = getBoxAxisSize(collider);
      } else if (collider.name == 'Capsule') {
        child = createCapsule(collider, mat, system.debug);
        axisSize = getCapsuleAxisSize(collider);
      } else if (collider.name == 'Plane') {
        child = createPlane(collider.plane, mat);
      } else if (collider.name == 'Sphere') {
        child = createSphere(collider, mat, system.debug);
        axisSize = getSphereAxisSize(collider);
      } else if (collider.name == 'HeightMap') {
        console.log('heightMap not implemented');
        return;
      } else if (collider.name == 'Mesh') {
        child = createMesh(collider, mat, system.debug);
        axisSize = getMeshAxisSize(collider);
      } else if ('clippedPlane' in collider) {
        console.log('clippedPlane not implemented');
        return;
      }
      if (collider.transform.rot) {
        child.quaternion.set(
            collider.transform.rot[1], collider.transform.rot[2],
            collider.transform.rot[3], collider.transform.rot[0]);
      }
      if (collider.transform.pos) {
        child.position.set(
            collider.transform.pos[0], collider.transform.pos[1],
            collider.transform.pos[2]);
      }
      if (system.debug && axisSize) {
        const debugAxis = new THREE.AxesHelper(axisSize);
        child.add(debugAxis);
        minAxisSize = Math.min(minAxisSize, axisSize);
      }
      parent.add(child);
    });
    scene.add(parent);
  });

  // Add contact position debug points.
  if (hasContactDebug(system)) {
    for (let i = 0; i < system.contact_pos[0].length; i++) {
      const parent = new THREE.Group();
      parent.name = 'contact' + i;
      let child;

      const mat = new THREE.MeshPhongMaterial({color: 0xff0000});
      const sphere_geom = new THREE.SphereGeometry(minAxisSize / 20.0, 6, 6);
      child = new THREE.Mesh(sphere_geom, mat);
      child.baseMaterial = child.material;
      child.castShadow = false;
      child.position.set(0, 0, 0);

      parent.add(child);
      scene.add(parent);
    }
  }

  return scene;
}

function createTrajectory(system) {
  const times = [...Array(system.pos.length).keys()].map(x => x * system.dt);
  const tracks = [];

  Object.entries(system.geoms).forEach(function(geom) {
    const name = geom[0];
    const i = geom[1][0].link_idx;
    if (i == null) {
      return;
    }
    const group = name.replaceAll('/', '_');  // sanitize node name
    const pos = system.pos.map(p => [p[i][0], p[i][1], p[i][2]]);
    const rot = system.rot.map(r => [r[i][1], r[i][2], r[i][3], r[i][0]]);
    tracks.push(new THREE.VectorKeyframeTrack(
        'scene/' + group + '.position', times, pos.flat()));
    tracks.push(new THREE.QuaternionKeyframeTrack(
        'scene/' + group + '.quaternion', times, rot.flat()));
  });

  // Add contact point debug.
  if (hasContactDebug(system)) {
    for (let i = 0; i < system.contact_pos[0].length; i++) {
      const group = 'contact' + i;
      const pos = system.contact_pos.map(p => [p[i][0], p[i][1], p[i][2]]);
      const visible = system.contact_penetration.map(p => p[i] > 1e-6);
      tracks.push(new THREE.VectorKeyframeTrack(
          'scene/' + group + '.position', times, pos.flat(),
          THREE.InterpolateDiscrete));
      tracks.push(new THREE.BooleanKeyframeTrack(
          'scene/' + group + '.visible', times, visible,
          THREE.InterpolateDiscrete));
    }
  }

  return new THREE.AnimationClip('Action', -1, tracks);
}

export {createScene, createTrajectory};
