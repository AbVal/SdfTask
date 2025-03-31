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
#include <cmath>

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

using std::min;
using std::max;

float EPS_GRAD = 1e-4f;
float EPS_MARCH = 1e-3f;
float MAX_MARCH_ITERATIONS = 75;
float MAX_MARCH_DIST = 12;
float T_MULT_CONST = 0.999f;

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
    bool reflected;
    Ray() {}
    Ray(float3 pos, float3 dir)
    {
        this->pos = pos;
        this->dir = dir;
    }
    Ray(float3 pos, float3 dir, bool reflected)
    {
        this->pos = pos;
        this->dir = dir;
        this->reflected = reflected;
    }
};

struct Material
{
    float reflection;
    float refraction;
    Material() {reflection = 0; refraction = 0;}
    Material(float reflection, float refraction)
    {
        this->reflection = reflection;
        this->refraction = refraction;
    }
};

struct Hit
{
    bool exist;
    float t;
    float4 color;
    Material material;
    float3 nVec;
    Hit() {exist = 0;}
    Hit(float t,  float4 color, Material material, float3 nVec)
    {
        this->exist = true;
        this->t = t;
        this->color = color;
        this->material = material;
        this->nVec = nVec;
    }
};

struct SceneObject
{
    float3 pos;
    float4 color;
    Material material;
    virtual float distance(float3 pos)
    {
        return 99999;
    }
    float3 normal(float3 pos)
    {
        float3 e1(EPS_GRAD, 0.0f, 0.0f);
        float3 e2(0.0f, EPS_GRAD, 0.0f);
        float3 e3(0.0f, 0.0f, EPS_GRAD);
        float dx = (this->distance(pos + e1) - this->distance(pos - e1)) / (2.0f * EPS_GRAD);
        float dy = this->distance(pos + e2) - this->distance(pos - e2) / (2.0f * EPS_GRAD);
        float dz = this->distance(pos + e3) - this->distance(pos - e3) / (2.0f * EPS_GRAD);

        return normalize(-float3(dx, dy, dz));
    }
    virtual Hit intersect(const Ray &ray)
    {
        return Hit();
    }
};

struct Sphere : SceneObject
{
    float radius;
    Sphere(float3 pos, float4 color, Material material, float radius)
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

    float3 normal(float3 pos)
    {
        return normalize(pos - this->pos);
    }

    Hit intersect(const Ray &ray) override
    {
        float3 k = ray.pos - this->pos;
        float b = dot(k, ray.dir);
        float c = dot(k, k) - this->radius * this->radius;
        float d = b * b - c;
      
        if(d >= 0)
        {
            float sqrtfd = sqrtf(d);
            float t1 = -b + sqrtfd;
            float t2 = -b - sqrtfd;
            float min_t  = min(t1,t2);
            float max_t = max(t1,t2);
            float t = (min_t >= 0) ? min_t : max_t;
            if (t > 0) {
                return Hit(t, this->color, this->material, this->normal(ray.pos + T_MULT_CONST * t * ray.dir));
            }
        
        }
        return Hit();
    }
};

float dot2(const float3 &x)
{
    return dot(x, x);
}

float clamp(const float &x, const float &a, const float &b)
{
    return min(max(x, a), b);
}


template <typename T> int sign(T val) {
    return (T(0) < val) - (val < T(0));
}

struct Triangle : SceneObject
{
    float3 a, b, c;
    float3 an, bn, cn;
    Triangle(float3 a, float3 b, float3 c, float4 color, Material material)
    {
        this->color = color;
        this->material = material;
        this->a = a;
        this->b = b;
        this->c = c;
    }
    Triangle(float3 a, float3 b, float3 c, float3 an, float3 bn, float3 cn, float4 color, Material material)
    {
        this->color = color;
        this->material = material;
        this->a = a;
        this->b = b;
        this->c = c;
        this->an = an;
        this->bn = bn;
        this->cn = cn;
    }

    float distance(float3 p) override
    {
        float3 ba = b - a; float3 pa = p - a;
        float3 cb = c - b; float3 pb = p - b;
        float3 ac = a - c; float3 pc = p - c;
        float3 nor = cross(ba, ac);
      
        return sqrt(
          (sign(dot(cross(ba, nor), pa)) +
           sign(dot(cross(cb, nor), pb)) +
           sign(dot(cross(ac, nor), pc)) < 2.0)
           ?
           min( min(
           dot2(ba * clamp(dot(ba, pa) / dot2(ba), 0.0, 1.0) - pa),
           dot2(cb * clamp(dot(cb, pb) / dot2(cb), 0.0, 1.0) - pb)),
           dot2(ac * clamp(dot(ac, pc) / dot2(ac), 0.0, 1.0) - pc))
           :
           dot(nor, pa)*dot(nor, pa) / dot2(nor));
      }

    Hit intersect(const Ray &ray) override
    {
        float3 v1v0 = b - a;
        float3 v2v0 = c - a;
        float3 rov0 = ray.pos - a;
        float3  n = cross(v1v0, v2v0);
        float3  q = cross(rov0, ray.dir);
        float d = 1.0/dot(ray.dir, n);
        float u = d*dot(-q, v2v0);
        float v = d*dot(q, v1v0);
        float t = d*dot(-n, rov0);
        if( u<0.0 || v<0.0 || (u+v)>1.0 ) return Hit(); //t = -1.0;
        return Hit(t, this->color, this->material, this->normal(ray.pos + T_MULT_CONST * t * ray.dir));
    }
};

struct Box : SceneObject
{
    float3 boxMin, boxMax;
    Box() {}
    Box(float3 boxMin, float3 boxMax, float4 color, Material material)
    {
        this->pos = (boxMin + boxMax) / 2;
        this->boxMin = boxMin;
        this->boxMax = boxMax;
        this->color = color;
        this->material = material;
    }

    float distance(float3 p) override
    {
        float3 q = abs(p) - this->pos;
        return length(clamp(q, 0.0f, 99999.0f)) + min(max(q.x, max(q.y, q.z)), 0.0f);
    }

    Hit intersect(const Ray &ray) override
    {
        float tmin, tmax;
        float3 inv_dir = 1.0f / ray.dir; 
        float lo = inv_dir.x * (boxMin.x - ray.pos.x);
        float hi = inv_dir.x * (boxMax.x - ray.pos.x);
        tmin = min(lo, hi);
        tmax = max(lo, hi);

        float lo1 = inv_dir.y * (boxMin.y - ray.pos.y);
        float hi1 = inv_dir.y * (boxMax.y - ray.pos.y);
        tmin = max(tmin, min(lo1, hi1));
        tmax = min(tmax, max(lo1, hi1));

        float lo2 = inv_dir.z * (boxMin.z - ray.pos.z);
        float hi2 = inv_dir.z * (boxMax.z - ray.pos.z);
        tmin = max(tmin, min(lo2, hi2));
        tmax = min(tmax, max(lo2, hi2));


        if ((tmin <= tmax) && (tmax > 0.f)) {
            return Hit(tmin, this->color, this->material, this->normal(ray.pos + T_MULT_CONST * tmin * ray.dir));
        }
        return Hit();
    }
};

struct Plane : SceneObject
{
    float a, b, c, d;
    float3 nVec;
    Plane(float3 pos, float4 color, Material material, float a, float b, float c, float d)
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

    Hit intersect(const Ray &ray) override
    {
        float t = -(dot(ray.pos, this->nVec) + this->d) / dot(ray.dir, this->nVec);
        if (t >= 0) {
            return Hit(t, this->color, this->material, this->normal(ray.pos + T_MULT_CONST * t * ray.dir));
        }
        return Hit();
    }
};

bool checkBounds(const float3 &point)
{
    if (point.x <= -1 || point.x >= 1) {
        return false;
    }
    if (point.y <= -1 || point.y >= 1) {
        return false;
    }
    if (point.z <= -1 || point.z >= 1) {
        return false;
    }
    return true;
}

bool checkGridBounds(const float3 &point, const SdfGrid &grid)
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
    return grid.data[x + y * grid.size.x + z * grid.size.x * grid.size.y];
}

float trilinearInterpolation(const float3 &point, const SdfGrid &grid)
{
    float3 point_scaled = (point + 1.0f) * float3(grid.size.x - 1, grid.size.y - 1, grid.size.z - 1) / 2.0f;
    if (!checkGridBounds(point_scaled, grid))
    {
        return 0.01f;
    }
    int3 lb = (int3) floor(point_scaled);
    int3 ub = (int3) ceil(point_scaled);
    ub.x += (ub.x == lb.x);
    ub.y += (ub.y == lb.y);
    ub.z += (ub.z == lb.z);
    float x, y, z;

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

Hit March(const Ray &ray, SceneObject& object)
{
    float3 start(ray.pos);
    float length = 0.0f;
    for (int i = 0; i < MAX_MARCH_ITERATIONS; i++)
    {
        float3 point = start + length * ray.dir;
        float distance = object.distance(point);
        length += distance;
        if (distance < EPS_MARCH) {
            
            return Hit(length, object.color, object.material, object.normal(start + T_MULT_CONST * length * ray.dir));
        }

        if (length > MAX_MARCH_DIST) break;
    }
    return Hit();
}

struct SdfGridObject : SceneObject
{
    SdfGrid grid;
    Box boundingBox;
    SdfGridObject(SdfGrid grid, float4 color, Material material)
    {
        this->grid = grid;
        this->color = color;
        this->material = material;
        this->boundingBox = Box(float3(-1.0f, -1.0f, -1.0f), float3(1.0f, 1.0f, 1.0f), float4(), Material());
    }

    float distance(float3 p) override
    {
        return trilinearInterpolation(p, this->grid);
    }

    Hit intersect(const Ray &ray) override
    {
        if (checkBounds(ray.pos))
        {
            return March(ray, *this);
        }
        Hit bboxHit = this->boundingBox.intersect(ray);
        if (bboxHit.exist)
        {
            Ray newRay(ray);
            newRay.pos += newRay.dir * (bboxHit.t + EPS_MARCH);
            if (!checkBounds(newRay.pos))
            {
                return Hit();
            }
            Hit innerHit = March(newRay, *this);
            if (innerHit.exist)
            {
                innerHit.t += bboxHit.t + EPS_MARCH;
                return innerHit;
            }
        }
        return Hit();
    }
};

struct LightingObject
{
    float3 pos;
    virtual float3 lightVec(float3 pos) {return float3(0, 1, 0);}
};

struct DirectionalLight : LightingObject
{
    float3 direction;
    DirectionalLight(float3 pos, float3 direction)
    {
        this->direction = normalize(direction);
    }
    float3 lightVec(float3 pos) override
    {
        return this->direction;
    }
};

Hit RaySceneIntersection(const Ray &ray, const std::vector<SceneObject*> &scene)
{
    float min_t = 999999999.0f;
    Hit bestHit;
    for (SceneObject* object : scene)
    {
        Hit objectHit = object[0].intersect(ray);
        if (!objectHit.exist)
            continue;
        if (objectHit.t < min_t) {
            min_t = objectHit.t;
            bestHit = objectHit;
        }
    }
    return bestHit;
}

float Shade(const float3 &nVec, const float3 &lightDir)
{
    return max(0.1f, dot(nVec, lightDir));
}

float3 Reflect(const float3& dir, const float3& nVec)

{
    float3 temp = nVec * dot(dir, nVec) * (-2);
    float3 rVec = normalize(temp + dir);
    return rVec;
}

float4 RayTrace(const Ray &ray, const std::vector<SceneObject*> &scene, const std::vector<LightingObject*> &lights)
{
    float4 color(0.0f, 0.0f, 0.0f, 0.0f);

    Hit hit = RaySceneIntersection(ray, scene);
    if (!hit.exist)
        return color;

    float3 hit_point = ray.pos + T_MULT_CONST * hit.t * ray.dir;
    // std::cout << hit_point.x << " " << hit_point.y << " " << hit_point.z << std::endl;

    color = hit.color;
    for (LightingObject* light : lights)
    {
        float3 lightDir = -light[0].lightVec(hit_point);
        Ray lightRay(hit_point, lightDir);
        Hit lightHit = RaySceneIntersection(lightRay, scene);
        if (lightHit.exist)
            color *= Shade(hit.nVec, lightDir);
        color *= Shade(hit.nVec, lightDir);
    }

    if (!ray.reflected && hit.material.reflection > 0)
    {
        Ray reflRay(hit_point, Reflect(ray.dir, hit.nVec), true);
        color += hit.material.reflection * RayTrace(reflRay, scene, lights);
    }


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

Ray CastRay(const AppData &app_data, int i, int j)
{
    float aspect_ratio = app_data.width / app_data.height;
    float u = (2.0f * (i + 0.5f) / app_data.width - 1.0f) * aspect_ratio; // * scale;
    float v = (2.0f * (j + 0.5f) / app_data.height - 1.0f); // * scale;
    Camera camera = app_data.camera;

    float3 dir = camera.u * u + camera.v * v + camera.camDir;

    return Ray(camera.pos, normalize(dir));

}

uint32_t colorRGBA(const float4 &color)
{
    int4 colorInt = (int4) clamp(color, 0, 255);
    return (((uint32_t) colorInt.x) << 24) + (((uint32_t) colorInt.y) << 16) + (((uint32_t) colorInt.z) << 8) + ((uint32_t) colorInt.w);
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
            float4 color = RayTrace(ray, app_data.scene, app_data.lighting);
            int pixel_idx = i + ((app_data.height - 1) - j) * app_data.width;
            pixels[pixel_idx] = colorRGBA(color);
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


void coutFloat4(const float4 &f)
{
    std::cout << f.x << ", " << f.y << ", " << f.z << ", " << f.w << std::endl;
}

float3 fromFloat4(const float4 &f)
{
    return float3(f.x, f.y, f.z);
}

std::vector<Triangle> ConvertToTriangles(const SimpleMesh &mesh) {
    std::vector<Triangle> triangles;
    const size_t triCount = mesh.TrianglesNum();
    
    for(size_t i = 0; i < triCount; ++i) {
        
        const uint32_t idx0 = mesh.indices[3*i + 0];
        const uint32_t idx1 = mesh.indices[3*i + 1];
        const uint32_t idx2 = mesh.indices[3*i + 2];
        
        float3 a = fromFloat4(mesh.vPos4f[idx0]);
        float3 an = fromFloat4(mesh.vNorm4f[idx0]);
        
        float3 b = fromFloat4(mesh.vPos4f[idx1]);
        float3 bn = fromFloat4(mesh.vNorm4f[idx1]);
        
        float3 c = fromFloat4(mesh.vPos4f[idx2]);
        float3 cn = fromFloat4(mesh.vNorm4f[idx2]);
        
        triangles.push_back(Triangle(a, b, c, an, bn, cn, float4(255.0, 255.0, 255.0, 255.0), Material()));
    }
    
    return triangles;
}


// SdfGrid sdfGridFromMesh(SimpleMesh &mesh, uint3 grid_size)
// {
// }

// You must include the command line parameters for your main function to be recognized by SDL
int main(int argc, char **args)
{
    // std::vector<Triangle> mesh_primitives;
    // SimpleMesh mesh;
    // if (argc > 1)
    // {
    //     std::cout << args[1] << std::endl;     
    //     if (strcmp(args[1], "conversion") == 0) {
    //         const char* input = args[2];
    //         const char* output = args[3];
    //         const char* mode = args[4];
    //         int param = std::stoi(args[5]);

    //         mesh = LoadMeshFromObj(input, true);
    //         // std::cout << mesh.indices.size() << std::endl;
    //         // mesh_primitives = ConvertToTriangles(mesh);
    //     }
    // }
    // exit(0);
    // const int SCREEN_WIDTH = 960;
    // const int SCREEN_HEIGHT = 960;
    const int SCREEN_WIDTH = 500;
    const int SCREEN_HEIGHT = 500;

    Camera camera(float3(-2.0, 0.0, -2.0), float3(1.0, 0, 1.0), float3(0.0, 1.0, 0.0));
    float speed = 0.025f;
    float sensitivity = 0.1f;
    updateCamWUV(camera);
    std::vector<SceneObject*> scene;
    // for (Triangle tr : mesh_primitives)
    // {
    //     scene.push_back(new Triangle(tr));
    // }
    // scene.push_back(new Plane(float3(0.0, 0.0, 0.0), float4(187.f,187.f,187.f,187.f), Material(0.75, 0), 0.1, 1, 0, 1));
    // scene.push_back(new Triangle(float3(0.0, 0.0, 0.0), float3(2.0, 0.0, 0.0), float3(0.0, 1.0, 2.0), float4(187.f,187.f,187.f,187.f), Material()));
    // scene.push_back(new Sphere(float3(2.0, 4.0, 1.0), float4(255.f,255.f,255.f,255.f), Material(), 0.25));
    // scene.push_back(new Sphere(float3(0.0, 1.0, 0.0), float4(255.f,25.f,25.f,25.f), Material(1, 0), 0.75));
    // scene.push_back(new Sphere(float3(0.1, 0.05, 0.0), float4(255.f,25.f,95.f,25.f), Material(1, 0), 0.2));
    // scene.push_back(new Sphere(float3(0.0, 0.0, 0.0), float4(187.f,187.f,187.f,255.f), Material(), 1.2));
    // scene.push_back(new Box(float3(-1.2, -1.2, -1.2), float3(-0.3, 0.02, -0.3), float4(255.f,255.f,255.f,255.f), Material()));
    // scene.push_back(new Box(float3(-0.5, -0.5, -0.5), float3(0.5, 0.5, 0.5), float4(255.f,255.f,255.f,255.f), Material()));
    // scene.push_back(new Box(float3(-1, -1, -1), float3(1, 1, 1), float4(255.f,255.f,255.f,255.f), Material()));
    // scene.push_back(new Box(float3(-0.8f, -0.8f, -0.8f), float3(0.8f, 0.8f, 0.8f), float4(255.f,0.f,0.f,255.f), Material()));
    // scene.push_back(new Sphere(float3(0.5, -0.25, 0.5), float4(187.f,187.f,187.f,255.f), Material(), 2));

    SdfGrid grid;
    load_sdf_grid(grid, "example_grid.grid");
    // SdfGrid grid = sdfGridFromMesh(mesh, uint3(10, 10, 10));
    scene.push_back(new SdfGridObject(grid, float4(255.f,255.f,255.f,255.f), Material()));

    std::vector<LightingObject*> lighting;
    // lighting.push_back(new DirectionalLight(float3(10.0, 10.0, 10.0), float3(0, -1.0, 0)));
    lighting.push_back(new DirectionalLight(float3(10.0, 10.0, 10.0), float3(1, -1.0, 0)));

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
        // std::cout << app_data.camera.pos.x << " " << app_data.camera.pos.y << " " << app_data.camera.pos.z << " " << std::endl;
        // std::cout << app_data.camera.camDir.x << " " << app_data.camera.camDir.y << " " << app_data.camera.camDir.z << " " << std::endl;
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
