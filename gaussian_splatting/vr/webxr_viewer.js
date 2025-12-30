// WebXR viewer for Gaussian Splatting in VR/AR

async function initWebXR(canvas, splatData, mode = 'immersive-vr') {
    if (!navigator.xr) {
        throw new Error('WebXR not supported');
    }

    const supported = await navigator.xr.isSessionSupported(mode);
    if (!supported) {
        throw new Error(`${mode} not supported`);
    }

    const gl = canvas.getContext('webgl2', { xrCompatible: true });

    const session = await navigator.xr.requestSession(mode, {
        requiredFeatures: ['local-floor'],
        optionalFeatures: ['hand-tracking', 'hit-test']
    });

    const baseLayer = new XRWebGLLayer(session, gl);
    await session.updateRenderState({ baseLayer });

    const refSpace = await session.requestReferenceSpace('local-floor');

    const state = {
        session,
        gl,
        refSpace,
        splatData,
        controllers: [],
        handedness: new Map()
    };

    setupInputSources(state);

    return state;
}

function setupInputSources(state) {
    state.session.addEventListener('inputsourceschange', (event) => {
        for (const source of event.added) {
            if (source.gripSpace) {
                state.controllers.push(source);
                state.handedness.set(source, source.handedness);
            }
        }

        for (const source of event.removed) {
            const idx = state.controllers.indexOf(source);
            if (idx >= 0) {
                state.controllers.splice(idx, 1);
                state.handedness.delete(source);
            }
        }
    });
}

function renderXRFrame(state, time, frame) {
    const session = state.session;
    const gl = state.gl;

    session.requestAnimationFrame((t, f) => renderXRFrame(state, t, f));

    const pose = frame.getViewerPose(state.refSpace);
    if (!pose) return;

    const layer = session.renderState.baseLayer;
    gl.bindFramebuffer(gl.FRAMEBUFFER, layer.framebuffer);

    for (const view of pose.views) {
        const viewport = layer.getViewport(view);
        gl.viewport(viewport.x, viewport.y, viewport.width, viewport.height);

        const viewMatrix = view.transform.inverse.matrix;
        const projMatrix = view.projectionMatrix;

        renderSplatsForView(state, viewMatrix, projMatrix);
    }

    renderControllers(state, frame);
}

function renderSplatsForView(state, viewMatrix, projMatrix) {
    const gl = state.gl;
    const splatData = state.splatData;

    gl.enable(gl.BLEND);
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
    gl.enable(gl.DEPTH_TEST);

    renderGaussianSplats(gl, splatData, viewMatrix, projMatrix);
}

function renderGaussianSplats(gl, splatData, viewMatrix, projMatrix) {
    const positions = new Float32Array(splatData.positions.flat());
    const colors = new Float32Array(splatData.colors.flat());
    const opacities = new Float32Array(splatData.opacities);

    const posBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, posBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, positions, gl.STATIC_DRAW);

    const colorBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, colorBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, colors, gl.STATIC_DRAW);

    gl.drawArrays(gl.POINTS, 0, splatData.positions.length);
}

function renderControllers(state, frame) {
    const gl = state.gl;

    for (const source of state.controllers) {
        const gripPose = frame.getPose(source.gripSpace, state.refSpace);

        if (gripPose) {
            const handedness = state.handedness.get(source);
            renderControllerModel(gl, gripPose.transform.matrix, handedness);
        }
    }
}

function renderControllerModel(gl, matrix, handedness) {
    const color = handedness === 'left' ? [0.3, 0.5, 1.0] : [1.0, 0.5, 0.3];

    gl.enable(gl.BLEND);
    renderSphere(gl, matrix, color, 0.05);
}

function renderSphere(gl, matrix, color, radius) {
    const position = [matrix[12], matrix[13], matrix[14]];

    const vertices = [];
    const segments = 16;

    for (let lat = 0; lat <= segments; lat++) {
        const theta = lat * Math.PI / segments;
        const sinTheta = Math.sin(theta);
        const cosTheta = Math.cos(theta);

        for (let lon = 0; lon <= segments; lon++) {
            const phi = lon * 2 * Math.PI / segments;
            const sinPhi = Math.sin(phi);
            const cosPhi = Math.cos(phi);

            const x = cosPhi * sinTheta;
            const y = cosTheta;
            const z = sinPhi * sinTheta;

            vertices.push(
                position[0] + radius * x,
                position[1] + radius * y,
                position[2] + radius * z
            );
        }
    }

    const buffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(vertices), gl.STATIC_DRAW);

    gl.drawArrays(gl.POINTS, 0, vertices.length / 3);
}

async function startVRSession(canvas, splatData) {
    const state = await initWebXR(canvas, splatData, 'immersive-vr');

    state.session.requestAnimationFrame((time, frame) => {
        renderXRFrame(state, time, frame);
    });

    return state;
}

async function startARSession(canvas, splatData) {
    const state = await initWebXR(canvas, splatData, 'immersive-ar');

    state.session.requestAnimationFrame((time, frame) => {
        renderXRFrame(state, time, frame);
    });

    return state;
}

function createVRButton(onSessionStart) {
    const button = document.createElement('button');
    button.textContent = 'Enter VR';
    button.style.cssText = `
        position: fixed;
        bottom: 20px;
        left: 50%;
        transform: translateX(-50%);
        padding: 12px 24px;
        background: #4a9eff;
        color: white;
        border: none;
        border-radius: 4px;
        font-size: 16px;
        cursor: pointer;
    `;

    button.addEventListener('click', onSessionStart);

    return button;
}

function createARButton(onSessionStart) {
    const button = document.createElement('button');
    button.textContent = 'Enter AR';
    button.style.cssText = `
        position: fixed;
        bottom: 20px;
        right: 20px;
        padding: 12px 24px;
        background: #ff9e4a;
        color: white;
        border: none;
        border-radius: 4px;
        font-size: 16px;
        cursor: pointer;
    `;

    button.addEventListener('click', onSessionStart);

    return button;
}

if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        initWebXR,
        startVRSession,
        startARSession,
        createVRButton,
        createARButton
    };
}
