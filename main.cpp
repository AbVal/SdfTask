#include "LiteMath/LiteMath.h"
#include "LiteMath/Image2d.h"

// stb_image is a single-header C library, which means one of your cpp files must have
//    #define STB_IMAGE_IMPLEMENTATION
//    #define STB_IMAGE_WRITE_IMPLEMENTATION
// since Image2d already defines the implementation, we don't need to do that here.
#include "stb_image.h"
#include "stb_image_write.h"

#include <SDL_keycode.h>
#include <cstdint>
#include <iostream>
#include <fstream>
#include <SDL.h>

#include <omp.h>
#include <chrono>

#include "mesh.h"
using namespace cmesh4;

using LiteMath::float2;
using LiteMath::float3;
using LiteMath::float4;
using LiteMath::float4x4;
using LiteMath::int2;
using LiteMath::int3;
using LiteMath::int4;
using LiteMath::uint2;
using LiteMath::uint3;
using LiteMath::uint4;

float EPS_GRAD = 1e-8;
float EPS_MARCH = 1e-3;
float MAX_MARCH_ITERATIONS = 75;
float MAX_MARCH_DIST = 12;

struct SdfGrid
{
    uint3 size;
    std::vector<float> data; // size.x*size.y*size.z values
};

void save_sdf_grid(const SdfGrid &scene, const std::string &path)
{
    std::ofstream fs(path, std::ios::binary);
    fs.write((const char *)&scene.size, 3 * sizeof(unsigned));
    fs.write((const char *)scene.data.data(), scene.size.x * scene.size.y * scene.size.z * sizeof(float));
    fs.flush();
    fs.close();
}

void load_sdf_grid(SdfGrid &scene, const std::string &path)
{
    std::ifstream fs(path, std::ios::binary);
    fs.read((char *)&scene.size, 3 * sizeof(unsigned));
    scene.data.resize(scene.size.x * scene.size.y * scene.size.z);
    fs.read((char *)scene.data.data(), scene.size.x * scene.size.y * scene.size.z * sizeof(float));
    fs.close();
}

struct SdfOctreeNode
{
    float values[8];
    unsigned offset; // offset for children (they are stored together). 0 offset means it's a leaf
};

struct SdfOctree
{
    std::vector<SdfOctreeNode> nodes;
};

void save_sdf_octree(const SdfOctree &scene, const std::string &path)
{
    std::ofstream fs(path, std::ios::binary);
    size_t size = scene.nodes.size();
    fs.write((const char *)&size, sizeof(unsigned));
    fs.write((const char *)scene.nodes.data(), size * sizeof(SdfOctreeNode));
    fs.flush();
    fs.close();
}

void load_sdf_octree(SdfOctree &scene, const std::string &path)
{
    std::ifstream fs(path, std::ios::binary);
    unsigned sz = 0;
    fs.read((char *)&sz, sizeof(unsigned));
    scene.nodes.resize(sz);
    fs.read((char *)scene.nodes.data(), scene.nodes.size() * sizeof(SdfOctreeNode));
    fs.close();
}

struct Camera
{
    float3 pos; //camera position
    float3 camDir; // direction
    float3 camUp;  // up
    Camera() {}
    Camera(float3 pos, float3 camDir, float3 camUp)
    {
        this->pos = pos;
        this->camDir = camDir;
        this->camUp = camUp;
    }
    float3 w; //-direction
    float3 u; //camUp x w
    float3 v; //w x u
};

void updateCamWUV(Camera &camera)
{
    camera.w = normalize(-camera.camDir);
    camera.u = normalize(cross(camera.camUp, camera.w));
    camera.v = normalize(cross(camera.w, camera.u));
}

#include <cmath> // For sinf, cosf, asinf

void updateCameraWithMouse(Camera &camera, float deltaX, float deltaY, float sensitivity) {
    // Apply yaw rotation around the camera's up vector (Y-axis)
    if (deltaX != 0.0f) {
        float yaw = deltaX * sensitivity;
        float cosYaw = cosf(yaw);
        float sinYaw = sinf(yaw);
        float3 originalDir = camera.camDir;

        float3 newDir;
        newDir.x = originalDir.x * cosYaw + originalDir.z * sinYaw;
        newDir.z = -originalDir.x * sinYaw + originalDir.z * cosYaw;
        newDir.y = originalDir.y;

        camera.camDir = normalize(newDir);
        updateCamWUV(camera); // Update u, v, w after yaw
    }

    // Apply pitch rotation around the camera's right vector (u)
    if (deltaY != 0.0f) {
        float pitch = deltaY * sensitivity;
        float cosP = cosf(pitch);
        float sinP = sinf(pitch);
        float3 originalDir = camera.camDir;

        // Rotate around the u vector using Rodrigues' rotation formula
        float3 rotatedDir = originalDir * cosP
                          + cross(camera.u, originalDir) * sinP
                          + camera.u * dot(camera.u, originalDir) * (1.0f - cosP);

        camera.camDir = normalize(rotatedDir);

        // Clamp pitch to prevent flipping
        const float maxPitch = 89.0f * (3.141592654f / 180.0f); // Convert to radians
        float currentPitch = asinf(camera.camDir.y);
        if (currentPitch > maxPitch) {
            camera.camDir.y = sinf(maxPitch);
            camera.camDir = normalize(camera.camDir);
        } else if (currentPitch < -maxPitch) {
            camera.camDir.y = sinf(-maxPitch);
            camera.camDir = normalize(camera.camDir);
        }

        updateCamWUV(camera); // Update u, v, w after pitch
    }
}

struct Ray
{
    float3 pos;
    float3 dir;
    Ray() {}
    Ray(float3 pos, float3 dir)
    {
        this->pos = pos;
        this->dir = dir;
    }
};

struct Material
{
    int reflection;
    int refraction;
    Material() {}
    Material(int reflection, int refraction)
    {
        this->reflection = reflection;
        this->refraction = refraction;
    }
};

struct Hit
{
    bool exist;
    float t;
    u_int32_t color;
    Material material;
    Hit() {exist = 0;}
    Hit(float t,  u_int32_t color, Material material)
    {
        this->exist = true;
        this->t = t;
        this->color = color;
        this->material = material;
    }
};

struct SceneObject
{
    float3 pos;
    uint32_t color;
    Material material;
    virtual float distance(float3 pos)
    {
        return 99999;
    }
    float3 normal(float3 pos)
    {
        float3 e1(EPS_GRAD, 0.0, 0.0);
        float3 e2(0.0, EPS_GRAD, 0.0);
        float3 e3(0.0, 0.0, EPS_GRAD);
        float dx = (this->distance(pos + e1) - this->distance(pos - e1)) / (2.0 * EPS_GRAD);
        float dy = this->distance(pos + e2) - this->distance(pos - e2) / (2.0 * EPS_GRAD);
        float dz = this->distance(pos + e3) - this->distance(pos - e3) / (2.0 * EPS_GRAD);

        return normalize(float3(dx, dy, dz));
    }
};

struct Sphere : SceneObject
{
    float radius;
    Sphere(float3 pos, uint32_t color, Material material, float radius)
    {
        this->pos = pos;
        this->color = color;
        this->material = material;
        this->radius = radius;
    }

    float distance(float3 p) override
    {
        return length(p - this->pos) - this->radius;
    }
};

struct Box : SceneObject
{
    float dist;
    Box(float3 pos, uint32_t color, Material material, float dist)
    {
        this->pos = pos;
        this->color = color;
        this->material = material;
        this->dist = dist;
    }

    float distance(float3 p) override
    {
        return hmax(p - this->pos) - this->dist;
    }
};

struct Plane : SceneObject
{
    float a, b, c, d;
    float3 nVec;
    Plane(float3 pos, uint32_t color, Material material, float a, float b, float c, float d)
    {
        this->pos = pos;
        this->color = color;
        this->material = material;
        this->a = a;
        this->b = b;
        this->c = c;
        this->d = d;
        this->nVec = normalize(float3(a, b, c));
    }

    float distance(float3 p) override
    {
        return abs(this->a * p.x + this->b * p.y + this->c * p.z + this->d) / length(this->nVec);
    }

    float3 normal(float3 pos)
    {
        return this->nVec;
    }
};

bool checkBounds(float3 point)
{
    if (point.x < -1 || point.x > 1) {
        return false;
    }
    if (point.y < -1 || point.y > 1) {
        return false;
    }
    if (point.z < -1 || point.z > 1) {
        return false;
    }
    return true;
}

bool checkGridBounds(float3 point, const SdfGrid &grid)
{
    if (point.x < 0 || point.x > grid.size.x - 1) {
        return false;
    }
    if (point.y < 0 || point.y > grid.size.y - 1) {
        return false;
    }
    if (point.z < 0 || point.z > grid.size.z - 1) {
        return false;
    }
    return true;
}

float gridIndex(const SdfGrid &grid, int x, int y, int z)
{
    // std::cout << "gridIndex" << std::endl;
    // std::cout << x << std::endl;
    // std::cout << y << std::endl;
    // std::cout << z << std::endl;
    return grid.data[x + y * grid.size.x + z * grid.size.x * grid.size.y];
}

float trilinearInterpolation(float3 point, const SdfGrid &grid)
{
    if (!checkBounds(point))
    {
        return hmax(abs(point)) - 1;
    }
    // std::cout << "trilinearInterpolation" << std::endl;
    float3 point_scaled = (point + 1) * float3(grid.size.x, grid.size.y, grid.size.z) / 2;
    if (!checkGridBounds(point_scaled, grid))
    {
        return 999;
    }
    int3 lb = (int3) floor(point_scaled);
    int3 ub = (int3) ceil(point_scaled);
    ub.x += (ub.x == lb.x);
    ub.y += (ub.y == lb.y);
    ub.z += (ub.z == lb.z);
    float x, y, z;
    
    // std::cout << point_scaled.x << std::endl;
    // std::cout << point_scaled.y << std::endl;
    // std::cout << point_scaled.z << std::endl;
    // std::cout << std::endl;
    // std::cout << std::endl;
    // std::cout << std::endl;


    x = (point_scaled.x - lb.x) / (ub.x - lb.x);
    y = (point_scaled.y - lb.y) / (ub.y - lb.y);
    z = (point_scaled.z - lb.z) / (ub.z - lb.z);
    float c00 = gridIndex(grid, lb.x, lb.y, lb.z) * (1 - x) + gridIndex(grid, ub.x, lb.y, lb.z) * x;
    float c01 = gridIndex(grid, lb.x, lb.y, ub.z) * (1 - x) + gridIndex(grid, ub.x, lb.y, ub.z) * x;
    float c10 = gridIndex(grid, lb.x, ub.y, lb.z) * (1 - x) + gridIndex(grid, ub.x, ub.y, lb.z) * x;
    float c11 = gridIndex(grid, lb.x, ub.y, ub.z) * (1 - x) + gridIndex(grid, ub.x, ub.y, ub.z) * x;
    float c0 = c00 * (1 - y) + c10 * y;
    float c1 = c01 * (1 - y) + c11 * y;
    return c0 * (1 - z) + c1 * z;
}

struct SdfGridObject : SceneObject
{
    SdfGrid grid;
    SdfGridObject(SdfGrid grid, uint32_t color, Material material)
    {
        this->grid = grid;
        this->color = color;
        this->material = material;
    }

    float distance(float3 p) override
    {
        return trilinearInterpolation(p, this->grid);
    }
};

struct LightingObject
{
    float3 pos;
    float3 lightVec(float3 pos);
};

struct DirectionalLight : LightingObject
{
    float3 direction;
    DirectionalLight(float3 pos, float3 direction)
    {
        this->direction = direction;
    }
    float3 lightVec(float3 pos)
    {
        return this->direction;
    }
};


Hit March(const Ray &ray, SceneObject& object)
{
    // std::cout << "March" << std::endl;
    float3 start(ray.pos);
    bool firstHit = true;
    if (hmax(abs(start)) < 1)
    {
        firstHit = false;
    }
    float length = 0.0;
    for (int i = 0; i < MAX_MARCH_ITERATIONS; i++)
    {
        float3 point = start + length * ray.dir;
        float distance = object.distance(point);
        if (distance < EPS_MARCH) {
            if (firstHit) {
                start += 1.2 * length * ray.dir;
                length = 0;
                firstHit = false;
                // length += 1e-3;
                continue;
            }
            else if (hmax(abs(point)) > 1) break;
            return Hit(length, object.color, object.material);
        }
        length += distance;
        if (length > MAX_MARCH_DIST) break;
    }
    return Hit();
}

Hit RaySceneIntersection(const Ray &ray, const std::vector<SceneObject*> &scene, const std::vector<LightingObject*> &lights)
{
    float min_t = 999999999.0;
    Hit bestHit;
    bestHit.exist = false;
    for (SceneObject* object : scene)
    {
        // TODO: move marching into object interface
        Hit objectHit = March(ray, object[0]);
        if (!objectHit.exist)
            continue;
        if (objectHit.t < min_t) {
            min_t = objectHit.t;
            bestHit = objectHit;
        }
    }
    return bestHit;
}

uint32_t RayTrace(const Ray &ray, const std::vector<SceneObject*> &scene, const std::vector<LightingObject*> &lights)
{
    uint32_t color = 0;

    Hit hit = RaySceneIntersection(ray, scene, lights);
    if (!hit.exist)
        return color;

    float3 hit_point = ray.pos + hit.t * ray.dir;
    color = hit.color;

    // for (int i = 0; i < NumLights; i++)
    //   if (Visible(hit_point, lights[i]))
    //     color += Shade(hit, lights[i]);

    // if (hit.material.reflection > 0)
    // {
    //   Ray reflRay = reflect(ray, hit);
    //   color += hit.material.reflection * RayTrace(reflRay);
    // }

    // if (hit.material.refraction > 0)
    // {
    //   Ray reflRay = refract(ray, hit);
    //   color += hit.material.refraction * RayTrace(reflRay);
    // }

    return color;
}


struct AppData
{
    int width;
    int height;
    Camera camera;
    std::vector<SceneObject*> scene;
    std::vector<LightingObject*> lighting;
};

void draw_sdf_grid_slice(const SdfGrid &grid, int z_level, int voxel_size,
                         int width, int height, std::vector<uint32_t> &pixels)
{
    constexpr uint32_t COLOR_EMPTY = 0xFF333333;  // dark gray
    constexpr uint32_t COLOR_FULL = 0xFFFFA500;   // orange
    constexpr uint32_t COLOR_BORDER = 0xFF000000; // black

    for (int y = 0; y < grid.size.y; y++)
    {
        for (int x = 0; x < grid.size.x; x++)
        {
            int index = x + y * grid.size.x + z_level * grid.size.x * grid.size.y;
            uint32_t color = grid.data[index] < 0 ? COLOR_FULL : COLOR_EMPTY;
            for (int i = 0; i <= voxel_size; i++)
            {
                for (int j = 0; j <= voxel_size; j++)
                {
                    // flip the y axis
                    int pixel_idx = (x * voxel_size + i) + ((height - 1) - (y * voxel_size + j)) * width;
                    if (i == 0 || i == voxel_size || j == 0 || j == voxel_size)
                        pixels[pixel_idx] = COLOR_BORDER;
                    else
                        pixels[pixel_idx] = color;
                }
            }
        }
    }
}

Ray CastRay(const AppData &app_data, int i, int j)
{
    float aspect_ratio = app_data.width / app_data.height;
    float u = (2.0f * (i + 0.5f) / app_data.width - 1.0f) * aspect_ratio; // * scale;
    float v = (2.0f * (j + 0.5f) / app_data.height - 1.0f); // * scale;
    Camera camera = app_data.camera;

    float3 dir = camera.u * u + camera.v * v + camera.camDir;

    return Ray(camera.pos, normalize(dir));

}

void draw_frame(const AppData &app_data, std::vector<uint32_t> &pixels)
{
    // your rendering code goes here
    #pragma omp parallel for num_threads(8) schedule(dynamic,1)
    for (int i = 0; i < app_data.width; i++)
    {
        for (int j = 0; j < app_data.height; j++)
        {
            Ray ray = CastRay(app_data, i, j);
            uint32_t color = RayTrace(ray, app_data.scene, app_data.lighting);
            int pixel_idx = i + ((app_data.height - 1) - j) * app_data.width;
            pixels[pixel_idx] = color;
        }
    }
}

void save_frame(const char *filename, const std::vector<uint32_t> &frame, uint32_t width, uint32_t height)
{
    LiteImage::Image2D<uint32_t> image(width, height, frame.data());

    // Convert from ARGB to ABGR
    for (uint32_t i = 0; i < width * height; i++)
    {
        uint32_t &pixel = image.data()[i];
        auto a = (pixel & 0xFF000000);
        auto r = (pixel & 0x00FF0000) >> 16;
        auto g = (pixel & 0x0000FF00);
        auto b = (pixel & 0x000000FF) << 16;
        pixel = a | b | g | r;
    }

    if (LiteImage::SaveImage(filename, image))
        std::cout << "Image saved to " << filename << std::endl;
    else
        std::cout << "Image could not be saved to " << filename << std::endl;
    // If you want a slightly more low-level API, You can manually do:
    //    stbi_write_png(filename, width, height, 4, (unsigned char*)frame.data(), width * 4)
}

// You must include the command line parameters for your main function to be recognized by SDL
int main(int argc, char **args)
{
    const int SCREEN_WIDTH = 960;
    const int SCREEN_HEIGHT = 960;
    // const int SCREEN_WIDTH = 400;
    // const int SCREEN_HEIGHT = 400;

    Camera camera(float3(-1.0, 0.0, -1.0), float3(1.0, 0, 1.0), float3(0.0, 1.0, 0.0));
    float speed = 0.01;
    float sensitivity = 0.1;
    updateCamWUV(camera);
    std::vector<SceneObject*> scene;
    // scene.push_back(new Plane(float3(0.0, 0.0, 0.0), 0xBBBBBBFF, Material(), 0.1, 1, 0, 1));
    // scene.push_back(new Sphere(float3(0.5, -0.25, 0.5), 0xFFFFFFFF, Material(), 0.25));
    // scene.push_back(new Sphere(float3(0.5, -0.25, 0.5), 0xBBBBBBFF, Material(), 2));

    SdfGrid grid;
    load_sdf_grid(grid, "example_grid.grid");
    scene.push_back(new SdfGridObject(grid, 0xFFFFFFFF, Material()));

    std::vector<LightingObject*> lighting;

    // Pixel buffer (RGBA format)
    std::vector<uint32_t> pixels(SCREEN_WIDTH * SCREEN_HEIGHT, 0xFFFFFFFF); // Initialize with white pixels

    AppData app_data;
    app_data.width = SCREEN_WIDTH;
    app_data.height = SCREEN_HEIGHT;
    app_data.camera = camera;
    app_data.scene = scene;
    app_data.lighting = lighting;

    // Initialize SDL. SDL_Init will return -1 if it fails.
    if (SDL_Init(SDL_INIT_EVERYTHING) < 0)
    {
        std::cerr << "Error initializing SDL: " << SDL_GetError() << std::endl;
        return 1;
    }

    // Create our window
    SDL_Window *window = SDL_CreateWindow("SDF Viewer", SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED,
                                          SCREEN_WIDTH, SCREEN_HEIGHT, SDL_WINDOW_SHOWN);

    // Make sure creating the window succeeded
    if (!window)
    {
        std::cerr << "Error creating window: " << SDL_GetError() << std::endl;
        return 1;
    }

    // Create a renderer
    SDL_Renderer *renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED);
    if (!renderer)
    {
        std::cerr << "Renderer could not be created! SDL_Error: " << SDL_GetError() << std::endl;
        SDL_DestroyWindow(window);
        SDL_Quit();
        return 1;
    }

    // Create a texture
    SDL_Texture *texture = SDL_CreateTexture(
        renderer,
        SDL_PIXELFORMAT_ARGB8888,    // 32-bit RGBA format
        SDL_TEXTUREACCESS_STREAMING, // Allows us to update the texture
        SCREEN_WIDTH,
        SCREEN_HEIGHT);

    if (!texture)
    {
        std::cerr << "Texture could not be created! SDL_Error: " << SDL_GetError() << std::endl;
        SDL_DestroyRenderer(renderer);
        SDL_DestroyWindow(window);
        SDL_Quit();
        return 1;
    }
    SDL_Event ev;
    bool running = true;
    float3 new_pos;
    float deltaX, deltaY;
    // Main loop
    int frameCount = 0;
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    while (running)
    {
        deltaX = 0;
        deltaY = 0;
        // Event loop
        while (SDL_PollEvent(&ev) != 0)
        {
            // check event type
            switch (ev.type)
            {
            case SDL_QUIT:
                // shut down
                running = false;
                break;
            case SDL_MOUSEMOTION:
                deltaX -= ev.motion.xrel;
                deltaY -= ev.motion.yrel;
                break;
            case SDL_KEYDOWN:
                // test keycode
                switch (ev.key.keysym.sym)
                {
                // W and S keys to change the slice of the grid currently rendered
                case SDLK_w:
                    new_pos = app_data.camera.pos + app_data.camera.camDir * speed;
                    app_data.camera.pos = new_pos;
                    break;
                case SDLK_s:
                    new_pos = app_data.camera.pos - app_data.camera.camDir * speed;
                    app_data.camera.pos = new_pos;
                    break;
                case SDLK_a:
                    new_pos = app_data.camera.pos - app_data.camera.u * speed;
                    app_data.camera.pos = new_pos;
                    break;
                case SDLK_d:
                    new_pos = app_data.camera.pos + app_data.camera.u * speed;
                    app_data.camera.pos = new_pos;
                    break;
                case SDLK_q:
                    new_pos = app_data.camera.pos + app_data.camera.camUp * speed;
                    app_data.camera.pos = new_pos;
                    break;
                case SDLK_e:
                    new_pos = app_data.camera.pos - app_data.camera.camUp * speed;
                    app_data.camera.pos = new_pos;
                    break;
                // ESC to exit
                case SDLK_ESCAPE:
                    running = false;
                    break;
                    // etc
                }
                break;
            }
        }
        updateCameraWithMouse(app_data.camera, deltaX, deltaY, 0.001);
        updateCamWUV(app_data.camera);
        // std::cout << app_data.camera.pos.x << std::endl;
        // std::cout << app_data.camera.pos.y << std::endl;
        // std::cout << app_data.camera.pos.z << std::endl;

        // Update pixel buffer
        draw_frame(app_data, pixels);

        // Update the texture with the pixel buffer
        SDL_UpdateTexture(texture, nullptr, pixels.data(), SCREEN_WIDTH * sizeof(uint32_t));

        // Clear the renderer
        SDL_RenderClear(renderer);

        // Copy the texture to the renderer
        SDL_RenderCopy(renderer, texture, nullptr, nullptr);

        // Update the screen
        SDL_RenderPresent(renderer);
        frameCount += 1;
        std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
        int seconds_elapsed = std::chrono::duration_cast<std::chrono::seconds>(end - begin).count();
        if (seconds_elapsed > 1)
        {
            char buff[20];
            snprintf(buff, sizeof(buff), "FPS: %d", frameCount / seconds_elapsed);
            SDL_SetWindowTitle(window, buff);
            begin = std::chrono::steady_clock::now();
            frameCount = 0;
        }

        SDL_Delay(10);
    }

    // Destroy the window. This will also destroy the surface
    SDL_DestroyWindow(window);

    // Quit SDL
    SDL_Quit();

    // End the program
    return 0;
}
