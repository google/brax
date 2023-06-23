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
  return capsule.length * 2;
}

function getSphereAxisSize(sphere) {
  return sphere.radius * 2;
}

function getBoxAxisSize(box) {
  return Math.max(box.halfsize[0], box.halfsize[1], box.halfsize[2]) * 4;
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

function createCylinder(radius, height, mat) {
  const geometry = new THREE.CylinderGeometry(radius, radius, height, 32);
  mat.side = THREE.DoubleSide;
  const cyl = new THREE.Mesh(geometry, mat);
  cyl.baseMaterial = cyl.material;
  cyl.castShadow = true;
  cyl.layers.enable(1);
  return cyl;
}

function createCapsule(capsule, mat) {
  const sphere_geom = new THREE.SphereGeometry(capsule.radius, 16, 16);
  const cylinder_geom = new THREE.CylinderGeometry(
      capsule.radius, capsule.radius, capsule.length);

  const sphere1 = new THREE.Mesh(sphere_geom, mat);
  sphere1.baseMaterial = sphere1.material;
  sphere1.position.set(0, 0, capsule.length / 2);
  sphere1.castShadow = true;
  sphere1.layers.enable(1);

  const sphere2 = new THREE.Mesh(sphere_geom, mat);
  sphere2.baseMaterial = sphere2.material;
  sphere2.position.set(0, 0, -capsule.length / 2);
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
  const geom = new THREE.BoxGeometry(
      2 * box.halfsize[0], 2 * box.halfsize[1], 2 * box.halfsize[2]);
  const mesh = new THREE.Mesh(geom, mat);
  mesh.castShadow = true;
  mesh.baseMaterial = mesh.material;
  mesh.layers.enable(1);
  return mesh;
}

function createPlane(plane, mat) {
  const geometry = new THREE.PlaneGeometry(2000, 2000);
  const mesh = new THREE.Mesh(geometry, mat);
  mesh.receiveShadow = true;
  mesh.baseMaterial = mesh.material;

  return mesh;
}

function createSphere(sphere, mat) {
  const geom = new THREE.SphereGeometry(sphere.radius, 16, 16);
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
  const meshGeoms = {};
  if (system.meshes) {
    Object.entries(system.meshes).forEach(function(geom) {
      meshGeoms[geom[0]] = geom[1];
    });
  }

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
      const mat = (collider.name == 'Plane') ?
          createCheckerBoard() :
          (collider.name == 'heightMap') ?
          new THREE.MeshStandardMaterial({color: color, flatShading: true}) :
          new THREE.MeshPhongMaterial({color: color});
      let child;
      let axisSize;
      if (collider.name == 'Box') {
        child = createBox(collider, mat);
        axisSize = getBoxAxisSize(collider);
      } else if (collider.name == 'Capsule') {
        child = createCapsule(collider, mat);
        axisSize = getCapsuleAxisSize(collider);
      } else if (collider.name == 'Plane') {
        child = createPlane(collider.plane, mat);
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
        child = createCylinder(collider.radius, collider.length, mat);
        axisSize = 2 * Math.max(collider.radius, collider.length);
      } else if ('clippedPlane' in collider) {
        console.log('clippedPlane not implemented');
        return;
      } else if (collider.name == 'Convex') {
        console.log('convex not implemented');
        return;
      }
      if (collider.transform.rot) {
        const quat = new THREE.Quaternion(
          collider.transform.rot[1], collider.transform.rot[2],
          collider.transform.rot[3], collider.transform.rot[0]);
        if (collider.name == 'Cylinder') {
          quat.multiply(qRotx90)
        }
        child.quaternion.fromArray(quat.toArray());
      }
      if (collider.transform.pos) {
        child.position.set(
            collider.transform.pos[0], collider.transform.pos[1],
            collider.transform.pos[2]);
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

  if (system.states.contact) {
    /* add contact point spheres  */
    for (let i = 0; i < system.states.contact.pos[0].length; i++) {
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
  const times =
      [...Array(system.states.x.pos.length).keys()].map(x => x * system.dt);
  const tracks = [];

  Object.entries(system.geoms).forEach(function(geom_tuple) {
    const name = geom_tuple[0];
    const geom = geom_tuple[1];
    const i = geom[0].link_idx;
    if (i == null) {
      return;
    }
    const group = name.replaceAll('/', '_');  // sanitize node name
    const pos = system.states.x.pos.map(p => [p[i][0], p[i][1], p[i][2]]);
    const rot =
        system.states.x.rot.map(r => [r[i][1], r[i][2], r[i][3], r[i][0]]);
    tracks.push(new THREE.VectorKeyframeTrack(
        'scene/' + group + '.position', times, pos.flat()));
    tracks.push(new THREE.QuaternionKeyframeTrack(
        'scene/' + group + '.quaternion', times, rot.flat()));
  });

  if (system.states.contact) {
    /* add contact debug point trajectory */
    for (let i = 0; i < system.states.contact.pos[0].length; i++) {
      const group = 'contact' + i;
      const pos = system.states.contact.pos.map(p => [p[i][0], p[i][1], p[i][2]]);
      const visible = system.states.contact.penetration.map(p => p[i] > 1e-6);
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
