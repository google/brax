import * as THREE from 'https://cdn.jsdelivr.net/gh/mrdoob/three.js@r135/build/three.module.js';
import 'https://cdn.jsdelivr.net/gh/mrdoob/three.js@r135/examples/jsm/geometries/ParametricGeometry.js';

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
  // make a checkerboard material
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
  const material = new THREE.MeshStandardMaterial( { map: texture } );

  // mesh
  const geometry = new THREE.PlaneGeometry( 2000, 2000);
  const mesh = new THREE.Mesh( geometry, material );
  mesh.rotation.x = -Math.PI / 2;
  mesh.receiveShadow = true;
  mesh.baseMaterial = mesh.material;

  return mesh;
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

  const geom = new ParametricGeometry(builder, n_subdiv, n_subdiv);
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

function createMesh(mesh, geom) {
  const bufferGeometry = new THREE.BufferGeometry();
  const vertices = geom.vertices;
  const positions = new Float32Array(vertices.length * 3);
  const scale = mesh.scale ? mesh.scale : 1;
  // Convert the coordinate system.
  vertices.forEach(function(vertice, i) {
      positions[i * 3] = vertice.x * scale;
      positions[i * 3 + 1] = vertice.z * scale;
      positions[i * 3 + 2] = vertice.y * scale;
  });
  const indices = new Uint16Array(geom.faces);
  for (let i = 1; i < indices.length; i += 3) {
      [indices[i + 1], indices[i]] = [indices[i], indices[i + 1]];
  }
  bufferGeometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  bufferGeometry.setIndex(new THREE.BufferAttribute(indices, 1));
  bufferGeometry.computeVertexNormals();

  const mesh3 = new THREE.Mesh(bufferGeometry, basicMaterial);
  mesh3.castShadow = true;
  mesh3.baseMaterial = mesh.material;
  return mesh3;
}

function addHat(child, collider) {
  const radius = collider.capsule.radius;

  const hat = new THREE.Group();
  hat.name = "santa hat";
  const hatRadius = radius * 0.95;
  const thickness = radius * 0.22;

  const points = [];
  for (let i = 0; i < 8; i++) {
      points.push(new THREE.Vector2(Math.cos(i * 0.2) * hatRadius, (i + 2) * radius / 4));
  }
  const beanie = new THREE.Mesh(new THREE.LatheGeometry(points), new THREE.MeshPhongMaterial({
      color: 0xff0000
  }));
  beanie.baseMaterial = beanie.material;
  hat.add(beanie);

  const whiteMaterial = new THREE.MeshPhongMaterial({
      color: 0xffffff
  });

  const pompom = new THREE.Mesh(new THREE.SphereGeometry(thickness, 8, 8), whiteMaterial);
  pompom.position.set(0, points[points.length - 1].y, 0);
  pompom.baseMaterial = pompom.material;
  hat.add(pompom);

  const side = new THREE.Mesh(new THREE.TorusGeometry(hatRadius, thickness, 8, 25), whiteMaterial);
  side.rotateX(Math.PI / 2);
  side.position.set(0, (hatRadius + thickness) / 2, 0);
  side.baseMaterial = side.material;
  hat.add(side);
  // Tilt the hat slightly.
  hat.rotateZ(0.3);

  const group = new THREE.Group();
  group.add(hat);
  group.add(child);
  return group;
}

function isHead(body, collider) {
  if (!('capsule' in collider)) {
    return false;
  }
  return ((body.name == 'torso' && collider.capsule.radius == 0.09) ||
      (body.name == '$ Torso' && collider.capsule.radius == 0.25));
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
      } else if ('mesh' in collider) {
        child = createMesh(collider.mesh, meshGeoms[collider.mesh.name]);
      }
      if (isHead(body, collider)) {
        child = addHat(child, collider);
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
