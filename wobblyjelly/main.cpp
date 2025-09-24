// wobblyjelly/main.cpp
// NOTE: You will likely need to adjust includes/names to your Hyprland source layout.
// This file demonstrates the approach: shader-based mesh displacement when dragging a window.

#include <cmath>
#include <vector>
#include <chrono>
#include <algorithm>
#include <iostream>
#include <any>

// Adjust path to where Hyprland exposes PluginAPI.hpp in your environment:
#include <hyprland/src/plugins/PluginAPI.hpp>

// OpenGL headers (may require linking glew/glad depending on build env)
#ifdef __APPLE__
#include <OpenGL/gl3.h>
#else
#include <GL/gl.h>
#endif

using namespace std::chrono;

inline HANDLE PHANDLE = nullptr;

struct JellyVertex {
    float x, y;
    float u, v;
};

struct WindowJellyState {
    bool dragging = false;
    float lastX = 0, lastY = 0;
    float vx = 0, vy = 0; // velocity
    double lastTime = 0;
    GLuint vao = 0, vbo = 0, ebo = 0;
    GLuint shader = 0;
    int gridCols = 20;
    int gridRows = 12;
    int indexCount = 0;
    // store texture id if available; if not, plugin must query it per render
    GLuint texId = 0;
};

// simple map from window pointer -> state
static std::unordered_map<void*, WindowJellyState> g_states;

// --- Shader sources ---
static const char* vertexShaderSrc = R"glsl(
#version 330 core
layout(location = 0) in vec2 aPos;
layout(location = 1) in vec2 aUV;

out vec2 vUV;

uniform vec2 uWinSize;
uniform vec2 uCenter;      // center of disturbance in window coords
uniform vec2 uVelocity;    // window movement velocity
uniform float uTime;
uniform float uAmplitude;
uniform float uFrequency;
uniform float uDamping;

void main() {
    // normalize position to [0..1]
    vec2 pos = (aPos + 0.5 * uWinSize) / uWinSize;
    // compute distance from disturbance center
    float dx = (aPos.x - uCenter.x);
    float dy = (aPos.y - uCenter.y);
    float dist = length(vec2(dx, dy));

    // radial wave: sin(freq*dist - time*speed) * amplitude * exp(-dist * damping)
    float wave = sin(uFrequency * dist - uTime * 6.2831) * uAmplitude * exp(-dist * uDamping);

    // also bias wave by velocity magnitude, so faster drag => bigger jelly
    float velMag = length(uVelocity);
    float disp = wave * (1.0 + velMag * 5.0);

    // displace vertex along normal perpendicular to surface (in screen space we push vertically & horizontally)
    vec2 displaced = aPos + normalize(vec2(dx + 0.0001, dy + 0.0001)) * disp;

    // convert to NDC [-1,1] assuming origin (0,0) at window center => need projection done by Hyprland's compositor
    // Here we assume Hyprland expects coordinates in screen pixel space; many compositors transform later.
    gl_Position = vec4((displaced.x / (uWinSize.x/2.0)), (displaced.y / (uWinSize.y/2.0)), 0.0, 1.0);
    vUV = aUV;
}
)glsl";

static const char* fragmentShaderSrc = R"glsl(
#version 330 core
in vec2 vUV;
out vec4 FragColor;

uniform sampler2D uTex;

void main() {
    FragColor = texture(uTex, vUV);
}
)glsl";

// --- helper: compile shader program ---
static GLuint compileShaderProgram(const char* vsSrc, const char* fsSrc) {
    auto compile = [](GLenum type, const char* src) -> GLuint {
        GLuint s = glCreateShader(type);
        glShaderSource(s, 1, &src, nullptr);
        glCompileShader(s);
        GLint ok;
        glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
        if (!ok) {
            char buf[1024]; glGetShaderInfoLog(s, 1024, nullptr, buf);
            std::cerr << "Shader compile error: " << buf << "\n";
        }
        return s;
    };

    GLuint vs = compile(GL_VERTEX_SHADER, vsSrc);
    GLuint fs = compile(GL_FRAGMENT_SHADER, fsSrc);
    GLuint prog = glCreateProgram();
    glAttachShader(prog, vs);
    glAttachShader(prog, fs);
    glLinkProgram(prog);
    GLint ok;
    glGetProgramiv(prog, GL_LINK_STATUS, &ok);
    if (!ok) {
        char buf[1024]; glGetProgramInfoLog(prog, 1024, nullptr, buf);
        std::cerr << "Program link error: " << buf << "\n";
    }
    glDeleteShader(vs); glDeleteShader(fs);
    return prog;
}

// generate grid mesh (centered at 0,0) with given cols/rows sized to winW, winH
static void generateGrid(WindowJellyState &st, int cols, int rows, float winW, float winH) {
    std::vector<JellyVertex> verts;
    std::vector<GLuint> idx;

    for (int r = 0; r <= rows; ++r) {
        for (int c = 0; c <= cols; ++c) {
            float tx = (float)c / (float)cols;
            float ty = (float)r / (float)rows;
            float px = (tx - 0.5f) * winW;
            float py = (ty - 0.5f) * winH;
            JellyVertex v;
            v.x = px;
            v.y = py;
            v.u = tx;
            v.v = 1.0f - ty; // uv flip if needed
            verts.push_back(v);
        }
    }

    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            int i0 = r*(cols+1) + c;
            int i1 = i0 + 1;
            int i2 = i0 + (cols+1);
            int i3 = i2 + 1;
            // two triangles
            idx.push_back(i0);
            idx.push_back(i2);
            idx.push_back(i1);

            idx.push_back(i1);
            idx.push_back(i2);
            idx.push_back(i3);
        }
    }

    // upload to GL
    if (!st.vao) glGenVertexArrays(1, &st.vao);
    if (!st.vbo) glGenBuffers(1, &st.vbo);
    if (!st.ebo) glGenBuffers(1, &st.ebo);

    glBindVertexArray(st.vao);
    glBindBuffer(GL_ARRAY_BUFFER, st.vbo);
    glBufferData(GL_ARRAY_BUFFER, verts.size() * sizeof(JellyVertex), verts.data(), GL_STATIC_DRAW);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, st.ebo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, idx.size() * sizeof(GLuint), idx.data(), GL_STATIC_DRAW);

    // attributes
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(JellyVertex), (void*)offsetof(JellyVertex, x));
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, sizeof(JellyVertex), (void*)offsetof(JellyVertex, u));

    glBindVertexArray(0);

    st.gridCols = cols;
    st.gridRows = rows;
    st.indexCount = (int)idx.size();
}

// --- Event callbacks ---

// move begin
static void onMoveBegin(void* self, SCallbackInfo& info, std::any data) {
    // data should be pointer to the window object (CWindow*)
    void* pWin = std::any_cast<void*>(data);
    auto &st = g_states[pWin];
    st.dragging = true;
    st.lastTime = duration_cast<duration<double>>(high_resolution_clock::now().time_since_epoch()).count();
    // initialize shader if needed
    if (!st.shader) st.shader = compileShaderProgram(vertexShaderSrc, fragmentShaderSrc);
}

// move update (continuous)
static void onMove(void* self, SCallbackInfo& info, std::any data) {
    void* pWin = std::any_cast<void*>(data);
    // Expected: info contains new window position; since we don't know exact types,
    // we'll try to extract a common pair if Hyprland passes them. Otherwise user adapts.
    // For now, we assume the window object has fields we can query in render phase.
    auto &st = g_states[pWin];
    double now = duration_cast<duration<double>>(high_resolution_clock::now().time_since_epoch()).count();
    double dt = std::max(1e-6, now - st.lastTime);
    st.lastTime = now;

    // here we can't get raw mouse pos generically; instead at actual render time we'll
    // compute velocity by comparing window position fields (see preRender hook).
    (void)info; (void)data;
}

// move end
static void onMoveEnd(void* self, SCallbackInfo& info, std::any data) {
    void* pWin = std::any_cast<void*>(data);
    auto &st = g_states[pWin];
    st.dragging = false;
    // leave jelly relax animation to decay naturally (velocity -> 0 due to damping)
}

// render hook (called per window)
static void onPreRenderWindow(void* self, SCallbackInfo& info, std::any data) {
    // data should be pointer/reference to window object (CWindow*)
    void* pWin = std::any_cast<void*>(data);

    // --- IMPORTANT:
    // The next lines depend on how Hyprland exposes the window structure.
    // Commonly you'd have something like:
    // CWindow* pWindow = (CWindow*)pWin;
    // auto winSize = pWindow->m_vRealSize; // Vector2D { x, y }
    // auto tex = pWindow->m_pTexture->m_tex; // or pWindow->m_pSurface->getTexture()
    //
    // Below are placeholders: replace with your actual field access.
    struct DummyWindow {
        float w, h;
        float xpos, ypos;
        GLuint tex;
    };
    // if you can cast, do: DummyWindow* pWindow = static_cast<DummyWindow*>(pWin);
    DummyWindow* pWindow = reinterpret_cast<DummyWindow*>(pWin); // placeholder

    if (!pWindow) return;

    auto &st = g_states[pWin];
    // lazy generate mesh sized to window
    if (!st.vao) {
        generateGrid(st, st.gridCols, st.gridRows, pWindow->w, pWindow->h);
    }

    // compute velocity using stored last pos (simple differentiation)
    double now = duration_cast<duration<double>>(high_resolution_clock::now().time_since_epoch()).count();
    double dt = std::max(1e-6, now - st.lastTime);
    // if lastX/Y == 0 maybe initialization; we try to keep previous pos
    float newX = pWindow->xpos;
    float newY = pWindow->ypos;
    if (st.lastTime == 0) {
        st.lastX = newX; st.lastY = newY;
    }
    st.vx = (newX - st.lastX) / dt;
    st.vy = (newY - st.lastY) / dt;
    st.lastX = newX; st.lastY = newY;
    st.lastTime = now;

    // choose center point of disturbance at current cursor relative to window center
    // For simplicity we use window center as center of wave:
    float centerX = 0.0f;
    float centerY = 0.0f;

    // configure shader uniforms
    glUseProgram(st.shader);
    GLint locWinSize = glGetUniformLocation(st.shader, "uWinSize");
    GLint locCenter = glGetUniformLocation(st.shader, "uCenter");
    GLint locVel = glGetUniformLocation(st.shader, "uVelocity");
    GLint locTime = glGetUniformLocation(st.shader, "uTime");
    GLint locAmp = glGetUniformLocation(st.shader, "uAmplitude");
    GLint locFreq = glGetUniformLocation(st.shader, "uFrequency");
    GLint locDamp = glGetUniformLocation(st.shader, "uDamping");

    float t = (float)now;
    glUniform2f(locWinSize, pWindow->w, pWindow->h);
    glUniform2f(locCenter, centerX, centerY);
    glUniform2f(locVel, st.vx, st.vy);
    glUniform1f(locTime, t);
    glUniform1f(locAmp, 6.0f);       // tweak amplitude
    glUniform1f(locFreq, 0.02f);     // tweak frequency
    glUniform1f(locDamp, 0.02f);     // tweak damping

    // bind the window texture. Replace with actual window texture handle
    GLuint tex = pWindow->tex;
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, tex);
    GLint locTex = glGetUniformLocation(st.shader, "uTex");
    glUniform1i(locTex, 0);

    // blend & state (depends on compositor expectations)
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    // draw grid
    glBindVertexArray(st.vao);
    glDrawElements(GL_TRIANGLES, st.indexCount, GL_UNSIGNED_INT, 0);
    glBindVertexArray(0);

    glUseProgram(0);
}

// Plugin init / exit
APICALL EXPORT PLUGIN_DESCRIPTION_INFO PLUGIN_INIT(HANDLE handle) {
    PHANDLE = handle;

    // register event handlers - names may differ in your Hyprland
    HyprlandAPI::addEventHandler(PHANDLE, "moveBegin", onMoveBegin);
    HyprlandAPI::addEventHandler(PHANDLE, "move", onMove);
    HyprlandAPI::addEventHandler(PHANDLE, "moveEnd", onMoveEnd);

    // register render hook (before window is drawn)
    HyprlandAPI::addRenderHook(PHANDLE, "preRenderWindow", onPreRenderWindow);

    return {
        .name = "Wobbly Jelly Move",
        .description = "Gives windows a jelly / wave deformation while dragging",
        .author = "neonora",
        .version = "0.2"
    };
}

APICALL EXPORT void PLUGIN_EXIT() {
    // cleanup GL resources
    for (auto &kv : g_states) {
        auto &st = kv.second;
        if (st.vbo) glDeleteBuffers(1, &st.vbo);
        if (st.ebo) glDeleteBuffers(1, &st.ebo);
        if (st.vao) glDeleteVertexArrays(1, &st.vao);
        if (st.shader) glDeleteProgram(st.shader);
    }
    g_states.clear();
}