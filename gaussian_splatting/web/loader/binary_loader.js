// Binary loader for streaming Gaussian splat data

async function loadBinarySplat(url) {
    const response = await fetch(url);
    const buffer = await response.arrayBuffer();

    return parseBinarySplat(buffer);
}

function parseBinarySplat(buffer) {
    const view = new DataView(buffer);
    let offset = 0;

    const numGaussians = view.getUint32(offset, true);
    offset += 4;

    const positions = [];
    const colors = [];
    const scales = [];
    const rotations = [];
    const opacities = [];

    for (let i = 0; i < numGaussians; i++) {
        positions.push([
            view.getFloat32(offset, true),
            view.getFloat32(offset + 4, true),
            view.getFloat32(offset + 8, true)
        ]);
        offset += 12;

        colors.push([
            view.getFloat32(offset, true),
            view.getFloat32(offset + 4, true),
            view.getFloat32(offset + 8, true)
        ]);
        offset += 12;

        scales.push([
            view.getFloat32(offset, true),
            view.getFloat32(offset + 4, true),
            view.getFloat32(offset + 8, true)
        ]);
        offset += 12;

        rotations.push([
            view.getFloat32(offset, true),
            view.getFloat32(offset + 4, true),
            view.getFloat32(offset + 8, true),
            view.getFloat32(offset + 12, true)
        ]);
        offset += 16;

        opacities.push(view.getFloat32(offset, true));
        offset += 4;
    }

    return { positions, colors, scales, rotations, opacities };
}

async function loadChunkedSplat(baseUrl, chunkIndices) {
    const chunks = await Promise.all(
        chunkIndices.map(idx => loadBinarySplat(`${baseUrl}/chunk_${idx}.bin`))
    );

    return mergeChunks(chunks);
}

function mergeChunks(chunks) {
    const merged = {
        positions: [],
        colors: [],
        scales: [],
        rotations: [],
        opacities: []
    };

    for (const chunk of chunks) {
        merged.positions.push(...chunk.positions);
        merged.colors.push(...chunk.colors);
        merged.scales.push(...chunk.scales);
        merged.rotations.push(...chunk.rotations);
        merged.opacities.push(...chunk.opacities);
    }

    return merged;
}

async function loadLODHierarchy(baseUrl) {
    const response = await fetch(`${baseUrl}/lod_metadata.json`);
    const metadata = await response.json();

    const lodLevels = {};

    for (let level = 0; level < metadata.num_levels; level++) {
        lodLevels[level] = await loadBinarySplat(`${baseUrl}/lod_${level}.bin`);
    }

    return { metadata, lodLevels };
}

function selectLODLevel(lodData, cameraDistance, screenResolution) {
    const numLevels = lodData.metadata.num_levels;

    const distanceThresholds = [2, 5, 10, 20];

    for (let i = 0; i < distanceThresholds.length; i++) {
        if (cameraDistance < distanceThresholds[i]) {
            return Math.min(i, numLevels - 1);
        }
    }

    return numLevels - 1;
}

if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        loadBinarySplat,
        loadChunkedSplat,
        loadLODHierarchy,
        selectLODLevel
    };
}
