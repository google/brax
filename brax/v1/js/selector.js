import * as THREE from 'https://cdn.jsdelivr.net/gh/mrdoob/three.js@r135/build/three.module.js';

class Selector extends THREE.EventDispatcher {
  constructor(viewer) {
    super();

    const ignore = ['target', 'ground', 'floor'];

    this.viewer = viewer;
    this.raycaster = new THREE.Raycaster();
    this.mousePos = new THREE.Vector2();
    this.selected = null;
    this.hovered = null;
    this.dragging = false;
    this.selectable = viewer.scene.children.filter(
        o =>
            o instanceof THREE.Group && !ignore.includes(o.name.toLowerCase()));

    const domElement = this.viewer.domElement;
    domElement.addEventListener('pointermove', this.onPointerMove.bind(this));
    domElement.addEventListener('pointerdown', this.onPointerDown.bind(this));
    domElement.addEventListener('pointerup', this.onPointerUp.bind(this));
  }

  onPointerMove(event) {
    event.preventDefault();
    this.dragging = true;

    const rect = this.viewer.domElement.getBoundingClientRect();
    this.mousePos.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
    this.mousePos.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
    this.raycaster.setFromCamera(this.mousePos, this.viewer.camera);
    const intersections =
        this.raycaster.intersectObjects(this.selectable, true);

    if (intersections.length > 0) {
      let object = intersections[0].object;
      while (object.parent && !object.name) {
        object = object.parent;
      }
      if (this.hovered !== object) {
        if (this.hovered) {
          this.dispatchEvent({type: 'hoveroff', object: this.hovered});
        }
        this.hovered = object;
        this.dispatchEvent({type: 'hoveron', object: this.hovered});
        this.viewer.domElement.style.cursor = 'pointer';
      }
    } else if (this.hovered !== null) {
      this.dispatchEvent({type: 'hoveroff', object: this.hovered});

      this.viewer.domElement.style.cursor = 'auto';
      this.hovered = null;
    }
  }

  onPointerDown(event) {
    event.preventDefault();
    this.dragging = false;
  }

  onPointerUp(event) {
    event.preventDefault();
    if (this.dragging) return;  // ignore drag events, only handle clicks
    this.raycaster.setFromCamera(this.mousePos, this.viewer.camera);
    const intersections =
        this.raycaster.intersectObjects(this.selectable, true);

    if (intersections.length > 0) {
      let object = intersections[0].object;
      while (object.parent && !object.name) {
        object = object.parent;
      }
      if (this.selected !== object) {
        if (this.selected) {
          this.dispatchEvent({type: 'deselect', object: this.selected});
        }
        this.selected = object;
        this.dispatchEvent({type: 'select', object: this.selected});
      }
    } else if (this.selected !== null) {
      this.dispatchEvent({type: 'deselect', object: this.selected});
      this.selected = null;
    }
  }
}

export {Selector};
