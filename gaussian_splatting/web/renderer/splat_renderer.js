// WebGL Gaussian Splatting Renderer

class SplatRenderer {
    constructor(canvas) {
        this.canvas = canvas;
        this.gl = canvas.getContext('webgl2');

        if (!this.gl) {
            throw new Error('WebGL2 not supported');
        }

        this.program = null;
        this.gaussianData = null;
        this.viewMatrix = this.createIdentityMatrix();
        this.projMatrix = this.createPerspectiveMatrix(60, canvas.width / canvas.height, 0.1, 100);
    }

    createIdentityMatrix() {
        return new Float32Array([
            1, 0, 0, 0,
            0, 1, 0, 0,
            0, 0, 1, 0,
            0, 0, 0, 1
        ]);
    }

    createPerspectiveMatrix(fov, aspect, near, far) {
        const f = 1.0 / Math.tan(fov * Math.PI / 360);
        const rangeInv = 1.0 / (near - far);

        return new Float32Array([
            f / aspect, 0, 0, 0,
            0, f, 0, 0,
            0, 0, (near + far) * rangeInv, -1,
            0, 0, near * far * rangeInv * 2, 0
        ]);
    }

    initShaders() {
        const vertexShaderSource = `#version 300 es
            precision highp float;

            in vec3 position;
            in vec3 color;
            in vec3 scale;
            in vec4 rotation;
            in float opacity;

            uniform mat4 viewMatrix;
            uniform mat4 projMatrix;

            out vec3 vColor;
            out float vOpacity;
            out vec2 vOffset;

            void main() {
                gl_Position = projMatrix * viewMatrix * vec4(position, 1.0);
                gl_PointSize = 20.0;

                vColor = color;
                vOpacity = opacity;
                vOffset = vec2(0.0);
            }
        `;

        const fragmentShaderSource = `#version 300 es
            precision highp float;

            in vec3 vColor;
            in float vOpacity;
            in vec2 vOffset;

            out vec4 fragColor;

            void main() {
                vec2 coord = gl_PointCoord - vec2(0.5);
                float dist = length(coord);

                if (dist > 0.5) {
                    discard;
                }

                float gaussian = exp(-4.0 * dist * dist);
                float alpha = vOpacity * gaussian;

                fragColor = vec4(vColor, alpha);
            }
        `;

        const vertexShader = this.compileShader(vertexShaderSource, this.gl.VERTEX_SHADER);
        const fragmentShader = this.compileShader(fragmentShaderSource, this.gl.FRAGMENT_SHADER);

        this.program = this.gl.createProgram();
        this.gl.attachShader(this.program, vertexShader);
        this.gl.attachShader(this.program, fragmentShader);
        this.gl.linkProgram(this.program);

        if (!this.gl.getProgramParameter(this.program, this.gl.LINK_STATUS)) {
            throw new Error('Program linking failed');
        }
    }

    compileShader(source, type) {
        const shader = this.gl.createShader(type);
        this.gl.shaderSource(shader, source);
        this.gl.compileShader(shader);

        if (!this.gl.getShaderParameter(shader, this.gl.COMPILE_STATUS)) {
            throw new Error('Shader compilation failed: ' + this.gl.getShaderInfoLog(shader));
        }

        return shader;
    }

    loadGaussians(data) {
        this.gaussianData = data;

        const positions = new Float32Array(data.positions.flat());
        const colors = new Float32Array(data.colors.flat());
        const scales = new Float32Array(data.scales.flat());
        const rotations = new Float32Array(data.rotations.flat());
        const opacities = new Float32Array(data.opacities);

        this.positionBuffer = this.createBuffer(positions);
        this.colorBuffer = this.createBuffer(colors);
        this.scaleBuffer = this.createBuffer(scales);
        this.rotationBuffer = this.createBuffer(rotations);
        this.opacityBuffer = this.createBuffer(opacities);
    }

    createBuffer(data) {
        const buffer = this.gl.createBuffer();
        this.gl.bindBuffer(this.gl.ARRAY_BUFFER, buffer);
        this.gl.bufferData(this.gl.ARRAY_BUFFER, data, this.gl.STATIC_DRAW);
        return buffer;
    }

    render() {
        this.gl.clear(this.gl.COLOR_BUFFER_BIT | this.gl.DEPTH_BUFFER_BIT);

        this.gl.useProgram(this.program);

        const viewMatrixLocation = this.gl.getUniformLocation(this.program, 'viewMatrix');
        const projMatrixLocation = this.gl.getUniformLocation(this.program, 'projMatrix');

        this.gl.uniformMatrix4fv(viewMatrixLocation, false, this.viewMatrix);
        this.gl.uniformMatrix4fv(projMatrixLocation, false, this.projMatrix);

        this.bindAttribute('position', this.positionBuffer, 3);
        this.bindAttribute('color', this.colorBuffer, 3);
        this.bindAttribute('scale', this.scaleBuffer, 3);
        this.bindAttribute('rotation', this.rotationBuffer, 4);
        this.bindAttribute('opacity', this.opacityBuffer, 1);

        this.gl.enable(this.gl.BLEND);
        this.gl.blendFunc(this.gl.SRC_ALPHA, this.gl.ONE_MINUS_SRC_ALPHA);
        this.gl.enable(this.gl.DEPTH_TEST);

        const numGaussians = this.gaussianData.positions.length;
        this.gl.drawArrays(this.gl.POINTS, 0, numGaussians);
    }

    bindAttribute(name, buffer, size) {
        const location = this.gl.getAttribLocation(this.program, name);
        this.gl.bindBuffer(this.gl.ARRAY_BUFFER, buffer);
        this.gl.enableVertexAttribArray(location);
        this.gl.vertexAttribPointer(location, size, this.gl.FLOAT, false, 0, 0);
    }

    setViewMatrix(matrix) {
        this.viewMatrix = matrix;
    }

    resize(width, height) {
        this.canvas.width = width;
        this.canvas.height = height;
        this.gl.viewport(0, 0, width, height);
        this.projMatrix = this.createPerspectiveMatrix(60, width / height, 0.1, 100);
    }
}

if (typeof module !== 'undefined' && module.exports) {
    module.exports = SplatRenderer;
}
