import * as THREE from 'https://threejs.org/build/three.module.js';

const basicMaterial = new THREE.MeshPhongMaterial({color: 0x665544});
const targetMaterial = new THREE.MeshPhongMaterial({color: 0xff2222});

function createCapsule(capsule) {
  const sphere_geom = new THREE.SphereGeometry(capsule.radius, 16, 16);
  const cylinder_geom = new THREE.CylinderGeometry(
      capsule.radius, capsule.radius, capsule.length - 2 * capsule.radius);

  const sphere1 = new THREE.Mesh(sphere_geom, basicMaterial);
  sphere1.baseMaterial = sphere1.material;
  sphere1.position.set(0, capsule.length / 2 - capsule.radius, 0);
  sphere1.castShadow = true;

  const sphere2 = new THREE.Mesh(sphere_geom, basicMaterial);
  sphere2.baseMaterial = sphere2.material;
  sphere2.position.set(0, -capsule.length / 2 + capsule.radius, 0);
  sphere2.castShadow = true;

  const cylinder = new THREE.Mesh(cylinder_geom, basicMaterial);
  cylinder.baseMaterial = cylinder.material;
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
  mesh.baseMaterial = mesh.material;
  return mesh;
}

function createPlane(plane) {
  const group = new THREE.Group();
  const mesh = new THREE.Mesh(
      new THREE.PlaneGeometry(2000, 2000),
      new THREE.MeshPhongMaterial({color: 0x999999, depthWrite: false}));
  mesh.rotation.x = -Math.PI / 2;
  mesh.receiveShadow = true;
  mesh.baseMaterial = mesh.material;
  group.add(mesh);

  const mesh2 = new THREE.GridHelper(2000, 2000, 0x000000, 0x000000);
  mesh2.material.opacity = 0.4;
  mesh2.material.transparent = true;
  mesh2.baseMaterial = mesh2.material;
  group.add(mesh2);
  return group;
}

function createSphere(sphere, name) {
  const geom = new THREE.SphereGeometry(sphere.radius, 16, 16);
  let mat = name.toLowerCase() == 'target' ? targetMaterial : basicMaterial;
  const mesh = new THREE.Mesh(geom, mat);
  mesh.castShadow = true;
  mesh.baseMaterial = mesh.material;
  return mesh;
}

function createHeightMap(heightMap) {
  const size = heightMap.size;
  const n_subdiv = Math.sqrt(heightMap.data.length) - 1;

  if (!Number.isInteger(n_subdiv)) {
    throw 'The data length for an height map should be a perfect square.';
  }

  function builder(v, u, target) {
    const idx = Math.round(v * (n_subdiv) + u * n_subdiv * (n_subdiv + 1));
    const x = u * size;
    const y = -v * size;
    const z = heightMap.data[idx];
    target.set(x, y, z).multiplyScalar(1);
  }

  const geom = new THREE.ParametricGeometry(builder, n_subdiv, n_subdiv);
  geom.normalizeNormals();

  const group = new THREE.Group();
  const mesh = new THREE.Mesh(
      geom,
      new THREE.MeshStandardMaterial({color: 0x796049, flatShading: true}));
  mesh.rotation.x = -Math.PI / 2;
  mesh.receiveShadow = true;
  group.add(mesh);
  return group;
}

function createScene(system) {
  const scene = new THREE.Scene();

  system.config.bodies.forEach(function(body) {
    const parent = new THREE.Group();
    parent.name = body.name.replaceAll('/', '_');  // sanitize node name
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
      } else if ('heightMap' in collider) {
        child = createHeightMap(collider.heightMap);
      }
      if (collider.rotation) {
        // convert from z-up to y-up coordinate system
        const rot = new THREE.Vector3(
            collider.rotation.x, collider.rotation.y, collider.rotation.z);
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
        child.position.set(
            collider.position.x, collider.position.z, collider.position.y);
      }
      parent.add(child);
    });
    scene.add(parent);
  });

  return scene;
}

function createTrajectory(system) {
  const times =
      [...Array(system.pos.length).keys()].map(x => x * system.config.dt);
  const tracks = [];

  system.config.bodies.forEach(function(body, bi) {
    const group = body.name.replaceAll('/', '_');  // sanitize node name
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

export {createScene, createTrajectory};
