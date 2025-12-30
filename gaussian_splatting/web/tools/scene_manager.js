// Scene management for Gaussian Splatting viewer

function createSceneManager() {
    const scenes = new Map();
    let activeScene = null;

    return {
        addScene,
        removeScene,
        setActiveScene,
        getActiveScene,
        listScenes,
        updateSceneMetadata
    };

    function addScene(name, splatData, metadata = {}) {
        const scene = {
            name,
            data: splatData,
            metadata: {
                createdAt: Date.now(),
                numGaussians: splatData.positions.length,
                ...metadata
            },
            bounds: computeBounds(splatData.positions),
            visible: true
        };

        scenes.set(name, scene);

        if (!activeScene) {
            activeScene = name;
        }

        return scene;
    }

    function removeScene(name) {
        scenes.delete(name);

        if (activeScene === name) {
            activeScene = scenes.size > 0 ? scenes.keys().next().value : null;
        }
    }

    function setActiveScene(name) {
        if (scenes.has(name)) {
            activeScene = name;
            return scenes.get(name);
        }
        return null;
    }

    function getActiveScene() {
        return activeScene ? scenes.get(activeScene) : null;
    }

    function listScenes() {
        return Array.from(scenes.entries()).map(([name, scene]) => ({
            name,
            numGaussians: scene.metadata.numGaussians,
            bounds: scene.bounds,
            visible: scene.visible
        }));
    }

    function updateSceneMetadata(name, metadata) {
        const scene = scenes.get(name);
        if (scene) {
            Object.assign(scene.metadata, metadata);
        }
    }
}

function computeBounds(positions) {
    const bounds = {
        min: [Infinity, Infinity, Infinity],
        max: [-Infinity, -Infinity, -Infinity]
    };

    for (const pos of positions) {
        for (let i = 0; i < 3; i++) {
            bounds.min[i] = Math.min(bounds.min[i], pos[i]);
            bounds.max[i] = Math.max(bounds.max[i], pos[i]);
        }
    }

    bounds.center = [
        (bounds.min[0] + bounds.max[0]) / 2,
        (bounds.min[1] + bounds.max[1]) / 2,
        (bounds.min[2] + bounds.max[2]) / 2
    ];

    bounds.size = [
        bounds.max[0] - bounds.min[0],
        bounds.max[1] - bounds.min[1],
        bounds.max[2] - bounds.min[2]
    ];

    return bounds;
}

function createMultiSceneRenderer(scenes) {
    const mergedData = {
        positions: [],
        colors: [],
        scales: [],
        rotations: [],
        opacities: []
    };

    for (const scene of scenes) {
        if (scene.visible) {
            mergedData.positions.push(...scene.data.positions);
            mergedData.colors.push(...scene.data.colors);
            mergedData.scales.push(...scene.data.scales);
            mergedData.rotations.push(...scene.data.rotations);
            mergedData.opacities.push(...scene.data.opacities);
        }
    }

    return mergedData;
}

function frustumCull(splatData, viewMatrix, projMatrix, bounds) {
    const culled = {
        positions: [],
        colors: [],
        scales: [],
        rotations: [],
        opacities: []
    };

    for (let i = 0; i < splatData.positions.length; i++) {
        const pos = splatData.positions[i];

        if (isInFrustum(pos, viewMatrix, projMatrix)) {
            culled.positions.push(pos);
            culled.colors.push(splatData.colors[i]);
            culled.scales.push(splatData.scales[i]);
            culled.rotations.push(splatData.rotations[i]);
            culled.opacities.push(splatData.opacities[i]);
        }
    }

    return culled;
}

function isInFrustum(position, viewMatrix, projMatrix) {
    const vp = multiplyMatrices(projMatrix, viewMatrix);

    const clipPos = transformPoint(vp, position);

    return Math.abs(clipPos[0]) <= clipPos[3] &&
           Math.abs(clipPos[1]) <= clipPos[3] &&
           clipPos[2] >= 0 && clipPos[2] <= clipPos[3];
}

function transformPoint(matrix, point) {
    const x = point[0], y = point[1], z = point[2];

    const w = matrix[3] * x + matrix[7] * y + matrix[11] * z + matrix[15];

    return [
        (matrix[0] * x + matrix[4] * y + matrix[8] * z + matrix[12]) / w,
        (matrix[1] * x + matrix[5] * y + matrix[9] * z + matrix[13]) / w,
        (matrix[2] * x + matrix[6] * y + matrix[10] * z + matrix[14]) / w,
        w
    ];
}

function multiplyMatrices(a, b) {
    const result = new Float32Array(16);

    for (let i = 0; i < 4; i++) {
        for (let j = 0; j < 4; j++) {
            let sum = 0;
            for (let k = 0; k < 4; k++) {
                sum += a[i * 4 + k] * b[k * 4 + j];
            }
            result[i * 4 + j] = sum;
        }
    }

    return result;
}

if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        createSceneManager,
        createMultiSceneRenderer,
        frustumCull
    };
}
