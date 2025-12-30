// Spatial audio for VR/AR Gaussian Splatting experiences

function createSpatialAudioContext() {
    const audioContext = new (window.AudioContext || window.webkitAudioContext)();
    const listener = audioContext.listener;

    const sources = new Map();

    return {
        audioContext,
        listener,
        createSource,
        updateListenerPosition,
        updateSourcePosition,
        removeSource
    };

    function createSource(id, audioBuffer, position, options = {}) {
        const source = audioContext.createBufferSource();
        source.buffer = audioBuffer;
        source.loop = options.loop || false;

        const panner = audioContext.createPanner();
        panner.panningModel = 'HRTF';
        panner.distanceModel = options.distanceModel || 'inverse';
        panner.refDistance = options.refDistance || 1;
        panner.maxDistance = options.maxDistance || 10000;
        panner.rolloffFactor = options.rolloffFactor || 1;
        panner.coneInnerAngle = options.coneInnerAngle || 360;
        panner.coneOuterAngle = options.coneOuterAngle || 360;
        panner.coneOuterGain = options.coneOuterGain || 0;

        if (listener.positionX) {
            panner.positionX.setValueAtTime(position[0], audioContext.currentTime);
            panner.positionY.setValueAtTime(position[1], audioContext.currentTime);
            panner.positionZ.setValueAtTime(position[2], audioContext.currentTime);
        } else {
            panner.setPosition(position[0], position[1], position[2]);
        }

        const gainNode = audioContext.createGain();
        gainNode.gain.value = options.volume || 1.0;

        source.connect(panner);
        panner.connect(gainNode);
        gainNode.connect(audioContext.destination);

        sources.set(id, {
            source,
            panner,
            gainNode,
            position: [...position]
        });

        return source;
    }

    function updateListenerPosition(position, forward, up) {
        if (listener.positionX) {
            listener.positionX.setValueAtTime(position[0], audioContext.currentTime);
            listener.positionY.setValueAtTime(position[1], audioContext.currentTime);
            listener.positionZ.setValueAtTime(position[2], audioContext.currentTime);

            listener.forwardX.setValueAtTime(forward[0], audioContext.currentTime);
            listener.forwardY.setValueAtTime(forward[1], audioContext.currentTime);
            listener.forwardZ.setValueAtTime(forward[2], audioContext.currentTime);

            listener.upX.setValueAtTime(up[0], audioContext.currentTime);
            listener.upY.setValueAtTime(up[1], audioContext.currentTime);
            listener.upZ.setValueAtTime(up[2], audioContext.currentTime);
        } else {
            listener.setPosition(position[0], position[1], position[2]);
            listener.setOrientation(forward[0], forward[1], forward[2], up[0], up[1], up[2]);
        }
    }

    function updateSourcePosition(id, position) {
        const sourceData = sources.get(id);
        if (!sourceData) return;

        const { panner } = sourceData;

        if (panner.positionX) {
            panner.positionX.setValueAtTime(position[0], audioContext.currentTime);
            panner.positionY.setValueAtTime(position[1], audioContext.currentTime);
            panner.positionZ.setValueAtTime(position[2], audioContext.currentTime);
        } else {
            panner.setPosition(position[0], position[1], position[2]);
        }

        sourceData.position = [...position];
    }

    function removeSource(id) {
        const sourceData = sources.get(id);
        if (sourceData) {
            sourceData.source.stop();
            sourceData.source.disconnect();
            sourceData.panner.disconnect();
            sourceData.gainNode.disconnect();
            sources.delete(id);
        }
    }
}

async function loadAudioFile(url) {
    const response = await fetch(url);
    const arrayBuffer = await response.arrayBuffer();

    const audioContext = new (window.AudioContext || window.webkitAudioContext)();
    const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);

    return audioBuffer;
}

function attachAudioToGaussian(audioContext, gaussianPosition, audioBuffer, options = {}) {
    const sourceId = `gaussian_${Date.now()}_${Math.random()}`;

    const source = audioContext.createSource(sourceId, audioBuffer, gaussianPosition, {
        loop: true,
        refDistance: 0.5,
        maxDistance: 10,
        rolloffFactor: 2,
        ...options
    });

    source.start(0);

    return sourceId;
}

function updateAudioFromCamera(audioContext, cameraTransform) {
    const position = [
        cameraTransform[12],
        cameraTransform[13],
        cameraTransform[14]
    ];

    const forward = [
        -cameraTransform[8],
        -cameraTransform[9],
        -cameraTransform[10]
    ];

    const up = [
        cameraTransform[4],
        cameraTransform[5],
        cameraTransform[6]
    ];

    audioContext.updateListenerPosition(position, forward, up);
}

function createAmbienceLayer(audioContext, audioBuffer, volume = 0.3) {
    const source = audioContext.audioContext.createBufferSource();
    source.buffer = audioBuffer;
    source.loop = true;

    const gainNode = audioContext.audioContext.createGain();
    gainNode.gain.value = volume;

    source.connect(gainNode);
    gainNode.connect(audioContext.audioContext.destination);

    source.start(0);

    return { source, gainNode };
}

if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        createSpatialAudioContext,
        loadAudioFile,
        attachAudioToGaussian,
        updateAudioFromCamera,
        createAmbienceLayer
    };
}
