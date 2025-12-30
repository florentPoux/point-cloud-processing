// Camera controls for Gaussian Splatting viewer

function createOrbitControls(canvas, camera) {
    const state = {
        isDragging: false,
        previousMousePosition: { x: 0, y: 0 },
        rotation: { x: 0, y: 0 },
        distance: 5.0,
        target: { x: 0, y: 0, z: 0 },
        pan: { x: 0, y: 0 }
    };

    canvas.addEventListener('mousedown', (e) => {
        state.isDragging = true;
        state.previousMousePosition = { x: e.clientX, y: e.clientY };
    });

    canvas.addEventListener('mouseup', () => {
        state.isDragging = false;
    });

    canvas.addEventListener('mousemove', (e) => {
        if (!state.isDragging) return;

        const deltaX = e.clientX - state.previousMousePosition.x;
        const deltaY = e.clientY - state.previousMousePosition.y;

        if (e.shiftKey) {
            state.pan.x -= deltaX * 0.01;
            state.pan.y += deltaY * 0.01;
        } else {
            state.rotation.y += deltaX * 0.01;
            state.rotation.x += deltaY * 0.01;
        }

        state.previousMousePosition = { x: e.clientX, y: e.clientY };
        updateCamera(camera, state);
    });

    canvas.addEventListener('wheel', (e) => {
        e.preventDefault();
        state.distance *= (1 + e.deltaY * 0.001);
        state.distance = Math.max(0.1, Math.min(100, state.distance));
        updateCamera(camera, state);
    });

    updateCamera(camera, state);

    return state;
}

function updateCamera(camera, state) {
    const viewMatrix = computeViewMatrix(
        state.rotation.x,
        state.rotation.y,
        state.distance,
        state.target,
        state.pan
    );

    camera.setViewMatrix(viewMatrix);
}

function computeViewMatrix(rotX, rotY, distance, target, pan) {
    const cx = Math.cos(rotX);
    const sx = Math.sin(rotX);
    const cy = Math.cos(rotY);
    const sy = Math.sin(rotY);

    const x = distance * cx * sy + pan.x;
    const y = distance * sx + pan.y;
    const z = distance * cx * cy;

    const eye = [x, y, z];
    const center = [target.x + pan.x, target.y + pan.y, target.z];
    const up = [0, 1, 0];

    return lookAt(eye, center, up);
}

function lookAt(eye, center, up) {
    const zAxis = normalize(subtract(eye, center));
    const xAxis = normalize(cross(up, zAxis));
    const yAxis = cross(zAxis, xAxis);

    return new Float32Array([
        xAxis[0], yAxis[0], zAxis[0], 0,
        xAxis[1], yAxis[1], zAxis[1], 0,
        xAxis[2], yAxis[2], zAxis[2], 0,
        -dot(xAxis, eye), -dot(yAxis, eye), -dot(zAxis, eye), 1
    ]);
}

function subtract(a, b) {
    return [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
}

function cross(a, b) {
    return [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0]
    ];
}

function dot(a, b) {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

function normalize(v) {
    const len = Math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
    return [v[0] / len, v[1] / len, v[2] / len];
}

if (typeof module !== 'undefined' && module.exports) {
    module.exports = { createOrbitControls };
}
