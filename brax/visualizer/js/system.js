import * as THREE from 'three';

function createCheckerBoard() {
  const width = 2;
  const height = 2;

  const size = width * height;
  const data = new Uint8Array(4 * size);
  const colors = [new THREE.Color(0x999999), new THREE.Color(0x888888)];

  for (let i = 0; i < size; i++) {
    const stride = i * 4;
    const ck = [0, 1, 1, 0];
    const color = colors[ck[i]];
    data[stride + 0] = Math.floor(color.r * 255);
    data[stride + 1] = Math.floor(color.g * 255);
    data[stride + 2] = Math.floor(color.b * 255);
    data[stride + 3] = 255;
  }
  const texture = new THREE.DataTexture(data, width, height, THREE.RGBAFormat);
  texture.wrapS = THREE.RepeatWrapping;
  texture.wrapT = THREE.RepeatWrapping;
  texture.repeat.set(1000, 1000);
  texture.needsUpdate = true;
  return new THREE.MeshPhongMaterial({map: texture});
}

function getCapsuleAxisSize(capsule) {
  return capsule.size[1] * 2;
}

function getSphereAxisSize(sphere) {
  return sphere.size[0] * 2;
}

function getBoxAxisSize(box) {
  return Math.max(box.size[0], box.size[1], box.size[2]) * 4;
}

/**
 * Gets an axis size for a mesh.
 * @param {!ObjType} geom a geometry object
 * @returns {!float} the axis size
 */
function getMeshAxisSize(geom) {
  let size = 0;
  for (let i = 0; i < geom.vert.length; i++) {
    let v = geom.vert[i];
    size = Math.max(v[0], v[1], v[2], size);
  }
  return size * 2;
}

function createCylinder(cylinder, mat) {
  const geometry = new THREE.CylinderGeometry(
      cylinder.size[0], cylinder.size[0], 2 * cylinder.size[1], 32);
  mat.side = THREE.DoubleSide;
  const cyl = new THREE.Mesh(geometry, mat);
  cyl.baseMaterial = cyl.material;
  cyl.castShadow = true;
  cyl.layers.enable(1);
  return cyl;
}

function createCapsule(capsule, mat) {
  const sphere_geom = new THREE.SphereGeometry(capsule.size[0], 16, 16);
  const cylinder_geom = new THREE.CylinderGeometry(
      capsule.size[0], capsule.size[0], 2 * capsule.size[1]);

  const sphere1 = new THREE.Mesh(sphere_geom, mat);
  sphere1.baseMaterial = sphere1.material;
  sphere1.position.set(0, 0, capsule.size[1]);
  sphere1.castShadow = true;
  sphere1.layers.enable(1);

  const sphere2 = new THREE.Mesh(sphere_geom, mat);
  sphere2.baseMaterial = sphere2.material;
  sphere2.position.set(0, 0, -capsule.size[1]);
  sphere2.castShadow = true;
  sphere2.layers.enable(1);

  const cylinder = new THREE.Mesh(cylinder_geom, mat);
  cylinder.baseMaterial = cylinder.material;
  cylinder.castShadow = true;
  cylinder.rotation.x = -Math.PI / 2;
  cylinder.layers.enable(1);

  const group = new THREE.Group();
  group.add(sphere1, sphere2, cylinder);
  return group;
}

function createBox(box, mat) {
  const geom =
      new THREE.BoxGeometry(2 * box.size[0], 2 * box.size[1], 2 * box.size[2]);
  const mesh = new THREE.Mesh(geom, mat);
  mesh.castShadow = true;
  mesh.baseMaterial = mesh.material;
  mesh.layers.enable(1);
  return mesh;
}

function createPlane(plane, mat) {
  let size;
  if (plane.size[0] == 0 && plane.size[1] == 0) {
    size = [2000, 2000];
  } else {
    size = plane.size;
  }
  const geometry = new THREE.PlaneGeometry(size[0], size[1]);
  const mesh = new THREE.Mesh(geometry, mat);
  mesh.receiveShadow = true;
  mesh.baseMaterial = mesh.material;

  return mesh;
}

function createSphere(sphere, mat) {
  const geom = new THREE.SphereGeometry(sphere.size[0], 16, 16);
  const mesh = new THREE.Mesh(geom, mat);
  mesh.castShadow = true;
  mesh.baseMaterial = mesh.material;
  mesh.layers.enable(1);
  return mesh;
}

function createMesh(meshGeom, mat) {
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
  mesh.layers.enable(1);
  return mesh;
}

function createScene(system) {
  const scene = new THREE.Scene();

  // Add a world axis for debugging.
  const worldAxis = new THREE.AxesHelper(100);
  const qRotx90 = new THREE.Quaternion(0.70710677, 0.0, 0.0, 0.7071067);
  worldAxis.visible = false;
  scene.add(worldAxis);

  let minAxisSize = 1e6;
  Object.entries(system.geoms).forEach(function(geom) {
    const name = geom[0];
    const parent = new THREE.Group();
    parent.name = name.replaceAll('/', '_');  // sanitize node name
    geom[1].forEach(function(collider) {
      const rgba = collider.rgba;
      const color = new THREE.Color(rgba[0], rgba[1], rgba[2]);
      let mat;
      if (collider.name == 'Plane' && collider.size[0] == 0 && collider.size[1] == 0) {
        mat = createCheckerBoard();
      } else if (collider.name == 'heightMap') {
        mat = new THREE.MeshStandardMaterial({color: color, flatShading: true});
      } else {
        mat = new THREE.MeshPhongMaterial({color: color});
      }
      let child;
      let axisSize;
      if (collider.name == 'Box') {
        child = createBox(collider, mat);
        axisSize = getBoxAxisSize(collider);
      } else if (collider.name == 'Capsule') {
        child = createCapsule(collider, mat);
        axisSize = getCapsuleAxisSize(collider);
      } else if (collider.name == 'Plane') {
        child = createPlane(collider, mat);
      } else if (collider.name == 'Sphere') {
        child = createSphere(collider, mat);
        axisSize = getSphereAxisSize(collider);
      } else if (collider.name == 'HeightMap') {
        console.log('heightMap not implemented');
        return;
      } else if (collider.name == 'Mesh') {
        child = createMesh(collider, mat);
        axisSize = getMeshAxisSize(collider);
      } else if (collider.name == 'Cylinder') {
        child = createCylinder(collider, mat);
        axisSize = 2 * Math.max(collider.size[0], collider.size[1]);
      }
      if (collider.rot) {
        const quat = new THREE.Quaternion(
            collider.rot[1], collider.rot[2], collider.rot[3], collider.rot[0]);
        if (collider.name == 'Cylinder') {
          quat.multiply(qRotx90)
        }
        child.quaternion.fromArray(quat.toArray());
      }
      if (collider.pos) {
        child.position.set(collider.pos[0], collider.pos[1], collider.pos[2]);
      }
      if (axisSize) {
        const debugAxis = new THREE.AxesHelper(axisSize);
        debugAxis.visible = false;
        child.add(debugAxis);
        minAxisSize = Math.min(minAxisSize, axisSize);
      }
      parent.add(child);
    });
    scene.add(parent);
  });

  return scene;
}

function createTrajectory(system) {
  const times = [...Array(system.states.x.length).keys()].map(
      x => x * system.opt.timestep);
  const tracks = [];

  Object.entries(system.geoms).forEach(function(geom_tuple) {
    const name = geom_tuple[0];
    const geom = geom_tuple[1];
    const i = geom[0].link_idx;
    if (i == null || i == -1) {
      return;
    }
    const group = name.replaceAll('/', '_');  // sanitize node name
    const pos = system.states.x.map(
        x => [x.pos[i][0], x.pos[i][1], x.pos[i][2]]);
    const rot =
        system.states.x.map(
            x => [x.rot[i][1], x.rot[i][2], x.rot[i][3], x.rot[i][0]]);
    tracks.push(new THREE.VectorKeyframeTrack(
        'scene/' + group + '.position', times, pos.flat()));
    tracks.push(new THREE.QuaternionKeyframeTrack(
        'scene/' + group + '.quaternion', times, rot.flat()));
  });

  return new THREE.AnimationClip('Action', -1, tracks);
}

export {createScene, createTrajectory};
