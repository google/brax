import * as THREE from 'https://cdn.jsdelivr.net/gh/mrdoob/three.js@r135/build/three.module.js';
import {ParametricGeometry} from 'https://cdn.jsdelivr.net/gh/mrdoob/three.js@r135/examples/jsm/geometries/ParametricGeometry.js';

function createCheckerBoard() {
    const width = 2;
    const height = 2;

    const size = width * height;
    const data = new Uint8Array( 3 * size );
    const colors = [new THREE.Color( 0x999999 ), new THREE.Color( 0x888888 )];

    for ( let i = 0; i < size; i ++ ) {
      const stride = i * 3;
      const ck = [0, 1, 1, 0];
      const color = colors[ck[i]];
      data[ stride + 0] = Math.floor( color.r * 255 );
      data[ stride + 1] = Math.floor( color.g * 255 );
      data[ stride + 2] = Math.floor( color.b * 255 );
    }
    const texture = new THREE.DataTexture( data, width, height, THREE.RGBFormat );
    texture.wrapS = THREE.RepeatWrapping;
    texture.wrapT = THREE.RepeatWrapping;
    texture.repeat.set( 1000, 1000 );
    return new THREE.MeshStandardMaterial( { map: texture } );
}

function createCapsule(capsule, mat) {
  const sphere_geom = new THREE.SphereGeometry(capsule.radius, 16, 16);
  const cylinder_geom = new THREE.CylinderGeometry(
      capsule.radius, capsule.radius, capsule.length - 2 * capsule.radius);

  const sphere1 = new THREE.Mesh(sphere_geom, mat);
  sphere1.baseMaterial = sphere1.material;
  sphere1.position.set(0, capsule.length / 2 - capsule.radius, 0);
  sphere1.castShadow = true;

  const sphere2 = new THREE.Mesh(sphere_geom, mat);
  sphere2.baseMaterial = sphere2.material;
  sphere2.position.set(0, -capsule.length / 2 + capsule.radius, 0);
  sphere2.castShadow = true;

  const cylinder = new THREE.Mesh(cylinder_geom, mat);
  cylinder.baseMaterial = cylinder.material;
  cylinder.castShadow = true;

  const group = new THREE.Group();
  group.add(sphere1, sphere2, cylinder);
  return group;
}

function createBox(box, mat) {
  const geom = new THREE.BoxBufferGeometry(
      2 * box.halfsize.x, 2 * box.halfsize.z, 2 * box.halfsize.y);
  const mesh = new THREE.Mesh(geom, mat);
  mesh.castShadow = true;
  mesh.baseMaterial = mesh.material;
  return mesh;
}

function createPlane(plane, mat) {
  const geometry = new THREE.PlaneGeometry( 2000, 2000);
  const mesh = new THREE.Mesh(geometry, mat);
  mesh.rotation.x = -Math.PI / 2;
  mesh.receiveShadow = true;
  mesh.baseMaterial = mesh.material;

  return mesh;
}

function createSphere(sphere, mat) {
  const geom = new THREE.SphereGeometry(sphere.radius, 16, 16);
  const mesh = new THREE.Mesh(geom, mat);
  mesh.castShadow = true;
  mesh.baseMaterial = mesh.material;
  return mesh;
}

function createHeightMap(heightMap, mat) {
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

  const geom = new ParametricGeometry(builder, n_subdiv, n_subdiv);
  geom.normalizeNormals();

  const group = new THREE.Group();
  const mesh = new THREE.Mesh(geom, mat);
  mesh.rotation.x = -Math.PI / 2;
  mesh.receiveShadow = true;
  group.add(mesh);
  return group;
}

function createMesh(mesh_config, geom, mat) {
  const bufferGeometry = new THREE.BufferGeometry();
  const vertices = geom.vertices;
  const positions = new Float32Array(vertices.length * 3);
  const scale = mesh_config.scale ? mesh_config.scale : 1;
  // Convert the coordinate system.
  vertices.forEach(function(vertice, i) {
      positions[i * 3] = vertice.x * scale;
      positions[i * 3 + 1] = vertice.z * scale;
      positions[i * 3 + 2] = -vertice.y * scale;
  });
  const indices = new Uint16Array(geom.faces);
  bufferGeometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  bufferGeometry.setIndex(new THREE.BufferAttribute(indices, 1));
  bufferGeometry.computeVertexNormals();

  const mesh = new THREE.Mesh(bufferGeometry, mat);
  mesh.castShadow = true;
  mesh.baseMaterial = mesh.material;
  return mesh;
}

function createScene(system) {
  const scene = new THREE.Scene();
  const meshGeoms = {};
  system.config.meshGeometries.forEach(function(geom) {
    meshGeoms[geom.name] = geom;
  });
  system.config.bodies.forEach(function(body) {
    const parent = new THREE.Group();
    parent.name = body.name.replaceAll('/', '_');  // sanitize node name
    body.colliders.forEach(function(collider) {
      const color = collider.color
        ? collider.color
        : body.name.toLowerCase() == 'target' ? '#ff2222' : '#665544';
      const mat = ('plane' in collider)
        ? createCheckerBoard()
        : ('heightMap' in collider)
          ? new THREE.MeshStandardMaterial({color: color, flatShading: true})
          : new THREE.MeshPhongMaterial({color: color});
      let child;
      if ('box' in collider) {
        child = createBox(collider.box, mat);
      } else if ('capsule' in collider) {
        child = createCapsule(collider.capsule, mat);
      } else if ('plane' in collider) {
        child = createPlane(collider.plane, mat);
      } else if ('sphere' in collider) {
        child = createSphere(collider.sphere, mat);
      } else if ('heightMap' in collider) {
        child = createHeightMap(collider.heightMap, mat);
      } else if ('mesh' in collider) {
        child = createMesh(collider.mesh, meshGeoms[collider.mesh.name], mat);
      }
      if (collider.rotation) {
        // convert from z-up to y-up coordinate system
        const rot = new THREE.Vector3(
            collider.rotation.x, collider.rotation.y, collider.rotation.z);
        rot.multiplyScalar(Math.PI / 180);
        const eul = new THREE.Euler();
        eul.setFromVector3(rot);
        child.quaternion.setFromEuler(eul);
        const tmp = child.quaternion.y;
        child.quaternion.y = child.quaternion.z;
        child.quaternion.z = -tmp;
      }
      if (collider.position) {
        child.position.set(
            collider.position.x, collider.position.z, -collider.position.y);
      }
      child.visible = !collider.hidden;
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
    const pos = system.pos.map(p => [p[bi][0], p[bi][2], -p[bi][1]]);
    const rot =
        system.rot.map(r => [r[bi][1], r[bi][3], -r[bi][2], r[bi][0]]);
    tracks.push(new THREE.VectorKeyframeTrack(
        'scene/' + group + '.position', times, pos.flat()));
    tracks.push(new THREE.QuaternionKeyframeTrack(
        'scene/' + group + '.quaternion', times, rot.flat()));
  });

  return new THREE.AnimationClip('Action', -1, tracks);
}

export {createScene, createTrajectory};
