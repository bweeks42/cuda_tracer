#include <curand.h>
#include <curand_kernel.h>
#include <random>
#include <chrono>
#include <stdio.h>
#include <iostream>
#include <fstream>

// Overloads
__host__ __device__ float3 operator+(const float3 a, const float3 b) {
    return {a.x + b.x, a.y + b.y, a.z + b.z};
}

__host__ __device__  float3 operator-(const float3& a, const float3& b) {
    return {a.x - b.x, a.y - b.y, a.z - b.z};
}

__host__ __device__ float3 operator*(const float3& a, const float3& b) {
    return {a.x * b.x, a.y * b.y, a.z * b.z};
}

__host__ __device__ inline float3 operator/(const float3& a, float b) {
    return {a.x / b, a.y / b, a.z / b};
}

__host__ __device__ inline float3 operator*(const float3& a, float b) {
    return {a.x * b, a.y * b, a.z * b};
}

__host__ __device__ inline float3 operator-(const float3& a) {
    return a*-1.0;
}

// Utils
__host__ __device__ inline float degrees_to_radians(float degrees) {
    return degrees * M_PI / 180.0;
}

__host__ __device__ inline float dot(const float3 u, const float3 v) {
    return u.x*v.x + u.y*v.y + u.z*v.z;
}
__host__ __device__ inline float3 cross(const float3 u, const float3 v) {
    return {
        u.y * v.z - u.z * v.y,
        u.z * v.x - u.x * v.z,
        u.x * v.y - u.y * v.x
    };
}

__host__ __device__ inline float length_squared(const float3 v) {
    return dot(v, v);
}

__host__ __device__ inline float length(const float3 v) {
    return sqrt(length_squared(v));
}

__host__ __device__ inline float3 unit_vector(const float3 v) {
    return v / length(v);
}

__device__ inline float random_float(float min, float max, int idx, curandState *global_state) {
    curandState local_state = global_state[idx];
    float rand = curand_uniform(&local_state);
    rand *= max - min+0.999999;
    rand += min;
    global_state[idx] = local_state;
    return rand;
}

__device__ inline float3 random_unit_in_disk(int idx, curandState *state) {
    float3 p;
    do {
        p = {random_float(-1.0, 1.0, idx, state), random_float(-1.0, 1.0, idx, state), 0.0};
        if (length_squared(p) >= 1.0) {continue;}
        return p;
    } while(true);
}

__device__ inline float3 random_unit_in_sphere(int idx, curandState *state) {
    float3 p;
    do {
        p = {
            random_float(-1.0, 1.0, idx, state), random_float(-1.0, 1.0, idx, state), random_float(-1.0, 1.0, idx, state)
        };
        if (length_squared(p) >= 1.0) {continue;}
        return p;
    } while(true);
}

__device__ inline float3 random_unit_vector(int idx, curandState *state) {
    return unit_vector(random_unit_in_sphere(idx, state));
}

__global__ void setup_kernel(int max_x, int max_y, curandState *state, unsigned long seed) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if ((i >= max_x) || (j >= max_y)) return;
    int pixel_index = j*max_x + i;
    curand_init(seed, pixel_index, 0, &state[pixel_index]);
}

enum MaterialType {
    Matte,
    Metal,
    Dialectric
};

struct Material {
    MaterialType mat_ty;
    float3 color;
    float fuzz;
    float refraction_index;
};

struct Sphere {
    float3 origin;
    float radius;
    Material material;
};

struct Ray {
    float3 origin;
    float3 direction;
};

__device__ inline float3 ray_at(const Ray *ray, float t) {
    return ray->origin + (ray->direction * t);
}


struct HitRecord {
    float t;
    float3 point;
    float3 normal;
    bool front_face;
    Material material;
    __device__ inline void set_face_normal(const Ray *ray, float3 outward_normal) {
        front_face = dot(ray->direction, outward_normal) < 0.0;
        if (front_face) {normal=outward_normal;} else {normal=-outward_normal;}
    }
};

struct ImageParams {
    float aspect_ratio;
    int image_width;
    int image_height;
    float3 origin;
    float3 look_at;
    float3 vup;
    float vfov;
    float distance_to_focus;
    float aperture;
};

struct Camera {
    float3 origin;
    float3 horizontal;
    float3 vertical;
    float3 llc;
    float lens_radius;
    float3 u;
    float3 v;
    float3 w;
};

Camera camera_from_params(ImageParams ip) {
    Camera c;
    float theta = degrees_to_radians(ip.vfov);
    float h = tan(theta/2.0);
    float view_height = 2.0 * h;
    float view_width = ip.aspect_ratio * view_height;
    float3 w = unit_vector(ip.origin - ip.look_at);
    float3 u = unit_vector(cross(ip.vup, w));
    float3 v = cross(w, u);

    c.origin = ip.origin;
    c.horizontal = u * view_width * ip.distance_to_focus;
    c.vertical = v * view_height * ip.distance_to_focus;
    c.llc = ip.origin - c.horizontal/2.0 - c.vertical/2.0 - w*ip.distance_to_focus;
    c.lens_radius = ip.aperture / 2.0;

    return c;
}

__device__ inline Ray get_ray(const Camera *c, float s, float t, int idx, curandState *state) {
    float3 rd = random_unit_in_disk(idx, state) * c->lens_radius;
    float3 offset = c->u * rd.x + c->v * rd.y;
    return Ray {
        c->origin + offset,
        c->llc + c->horizontal*s + c->vertical*t - c->origin - offset
    };
}

__device__ inline bool hit_sphere(Sphere *sphere, const Ray *ray, float min, float max, HitRecord *hit) {
    float3 oc = ray->origin - sphere->origin;
    float a = length_squared(ray->direction);
    float half_b = dot(oc, ray->direction);
    float c = length_squared(oc) - (sphere->radius * sphere->radius);
    float discriminant = half_b*half_b - a*c;
    if (discriminant < 0.0) {
        return false;
    }
    float sqrtd = sqrt(discriminant);
    float root = (-half_b - sqrtd) / a;
    if (root < min || max < root) {
        root = (-half_b + sqrtd) / a;
        if (root < min || max < root) {
            return false;
        }
    }

    hit->t = root;
    hit->point = ray_at(ray, hit->t);
    float3 outward_normal = (hit->point - sphere->origin) / sphere->radius;
    hit->set_face_normal(ray, outward_normal);
    hit->material = sphere->material;
    return true;
}

__device__ inline bool hit_in_list(const Sphere *spheres, const int sphere_count, const Ray *ray, float min, float max, HitRecord *hit) {
    bool did_hit = false;
    float closest = max;
    HitRecord temp;
    int sphere_index = 0;
    while(sphere_index < sphere_count) {
        Sphere s = spheres[sphere_index];
        if (hit_sphere(&s, ray, min, closest, &temp)) {
            did_hit = true;
            closest = temp.t;
            *hit = temp;
        }
        sphere_index += 1;
    }
    return did_hit;
}

__device__ inline bool near_zero(float3 v) {
    float s = 1e-8;
    return abs(v.x) < s && abs(v.y) < s && abs(v.z) < s;
}

__device__ inline float3 reflect(float3 u, float3 v) {
    return u - v * 2.0 * dot(u, v);
}

__device__ inline float3 refract(float3 uv, float3 n, float etai_over_etat) {
    float cos_theta  = min(dot(-uv, n), 1.0);
    float3 r_out_perp = (uv + (n*cos_theta)) * etai_over_etat;
    float perp_2 = abs(1.0 - length_squared(r_out_perp));
    float3 r_out_parl = n * -sqrt(perp_2);
    return r_out_perp + r_out_parl;
}

__device__ inline float reflectance(float cos_theta, float refraction_ratio) {
    float r0 = (1.0 - refraction_ratio) / (1.0 + refraction_ratio);
    r0 = r0*r0;
    return r0 + (1.0 - r0) * pow((1.0 - cos_theta), 5.0);
}

__device__ inline bool scatter_for_material(const Ray *ray_in, const HitRecord *hit, float3 *scatter_color, Ray *scatter, int idx, curandState *state) {
    if (hit->material.mat_ty == Matte) {
        float3 scatter_direction = hit->normal + random_unit_vector(idx, state);
        if (near_zero(scatter_direction)) {
            scatter_direction = hit->normal;
        }
        scatter->origin = hit->point;
        scatter->direction = scatter_direction;
        *scatter_color = hit->material.color;
        return true;
    } else if (hit->material.mat_ty == Metal) {
        float3 reflected = reflect(unit_vector(ray_in->direction), hit->normal);
        scatter->origin = hit->point;
        scatter->direction = reflected + random_unit_in_sphere(idx, state)*hit->material.fuzz;
        *scatter_color = hit->material.color;
        return dot(scatter->direction, hit->normal) > 0.0;
    } else if (hit->material.mat_ty == Dialectric) {
        *scatter_color = {1.0, 1.0, 1.0};
        float refraction_ratio;
        if (hit->front_face) {refraction_ratio = 1.0/hit->material.refraction_index;} else {refraction_ratio = hit->material.refraction_index;}
        float3 unit_direction = unit_vector(ray_in->direction);
        float cos_theta = min(dot(-unit_direction, hit->normal), 1.0);
        float sin_theta = sqrt(1.0 - cos_theta*cos_theta);
        bool cannot_reflect = (refraction_ratio * sin_theta) > 1.0;
        float3 direction;
        if (cannot_reflect || reflectance(cos_theta, refraction_ratio) > random_float(0.0, 1.0, idx, state)) {
            direction = reflect(unit_direction, hit->normal);
        } else {
            direction = refract(unit_direction, hit->normal, refraction_ratio);
        }
        scatter->origin = hit->point;
        scatter->direction = direction;
        return true;
    }
    return false;
}

__device__ inline float3 ray_color(const Ray *ray, const Sphere *spheres, const int sphere_count, const int max_depth, int idx, curandState *state) {
    float3 color = {1.0, 1.0, 1.0};
    int depth = 0;
    Ray cur_ray = *ray;
    while(depth < max_depth) {
        HitRecord hit;
        if (hit_in_list(spheres, sphere_count, &cur_ray, 0.001, (float)1e8, &hit)) {
            Ray scatter = {{0.0, 0.0, 0.0}, 0.0};
            float3 scatter_color = {0.0, 0.0, 0.0};
            if (scatter_for_material(&cur_ray, &hit, &scatter_color, &scatter, idx, state)) {
                color = color * scatter_color;
                cur_ray = scatter;
                depth += 1;
                continue;
            }
            color = color * 0.0;
            break;
        }
        float3 unit_direction = unit_vector(cur_ray.direction); // TODO: Is this cur_ray or ray?
        float t = 0.5 * (unit_direction.y + 1.0);
        float3 a = {1.0, 1.0, 1.0};
        a = a * (1.0 - t);
        float3 b = {0.5, 0.7, 1.0};
        b = b * t;
        color = color * (a+b);
        break;
    }
    return color;
}

__device__ float clamp(float value, float min, float max) {
    return fmaxf(min, fminf(value, max));
}

__global__ void ray_trace(Sphere *spheres, int sphere_count, float3 *pixels, int max_x, int max_y, Camera c, int image_height, int image_width, curandState *global_state) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if ((i >= max_x) || (j >= max_y)) return;
    int idx = j*max_x + i;
    int max_depth = 50;
    int inv_j = image_height - 1 - j;
    int n_samples = 40;
    float3 final_color = {0.0, 0.0, 0.0};
    int sample = 0;
    float scale = 1.0/(float)n_samples;

    while(sample < n_samples) {
        float u = ((float)i + random_float(0.0, 1.0, idx, global_state)) / (float)(image_width - 1);
        float v = ((float)inv_j +random_float(0.0, 1.0, idx, global_state)) / (float)(image_height -1);
        Ray ray = get_ray(&c, u, v, idx, global_state);
        final_color = final_color + ray_color(&ray, spheres, sphere_count, max_depth, idx, global_state);
        sample += 1;
    }
    float r = sqrt(final_color.x * scale);
    float g = sqrt(final_color.y * scale);
    float b = sqrt(final_color.z * scale);
    float ir = (256.0 * clamp(r, 0.0, 0.999));
    float ib = (256.0 * clamp(g, 0.0, 0.999));
    float ig = (256.0 * clamp(b, 0.0, 0.999));



    pixels[idx] = {ir, ib, ig};
}

float random_float_in_range(float min, float max) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(min, max);

    return dis(gen);
}

int main() {

    // Setup image parameters
    ImageParams params;
    params.aspect_ratio = 3.0/2.0;
    params.image_width = pow(2, 12);
    params.image_height = (int)((float)params.image_width/params.aspect_ratio);
    params.origin = {13.0, 2.0, 3.0};
    params.look_at = {0.0, 0.0, 0.0};
    params.vup = {0.0, 1.0, 0.0};
    params.vfov = 20.0;
    params.distance_to_focus = 10.0;
    params.aperture = 1.0;

    Camera c = camera_from_params(params);

    // Pixel buffer
    int pixel_count = params.image_width * params.image_height;
    float3 *d_pixels;
    cudaMalloc(&d_pixels, pixel_count*sizeof(float3));


    printf("Generating image of width: %d, height: %d total pixels: %d\n", params.image_width, params.image_height, pixel_count);

    // setup rand state
    curandState *rand_states;
    cudaMalloc(&rand_states, pixel_count * sizeof(curandState));
    srand(time(0));
    int seed = rand();

    // setup spheres
    printf("Generating spheres...\n");

    // Setup ground
    Material ground_material = {
        .mat_ty = Matte,
        .color = {0.5, 0.5, 0.5},
        .fuzz = 0.0,
        .refraction_index = 0.0
    };
    Sphere ground = {
        .origin = {0.0, -1000.0, 0.0},
        .radius = 1000.0,
        .material = ground_material
    };
    
    // Matte sphere
    Material matte_material = {
        .mat_ty = Matte,
        .color = {0.4, 0.2, 0.1},
        .fuzz = 0.0,
        .refraction_index = 0.0
    };
    Sphere matte_sphere = {
        .origin = {-4.0, 1.0, 0.0},
        .radius = 1.0,
        .material = matte_material
    };

    // Glass sphere
    Material glass_material = {
        .mat_ty = Dialectric,
        .color = {0.0, 0.0, 0.0},
        .fuzz = 0.0,
        .refraction_index = 1.5
    };

    Sphere glass_sphere = {
        .origin = {0.0, 1.0, 0.0},
        .radius = 1.0,
        .material = glass_material
    };


    // Metal sphere
    Material metal_material = {
        .mat_ty = Metal,
        .color = {0.7, 0.6, 0.5},
        .fuzz = 0.0,
        .refraction_index = 0.0
    };
    Sphere metal_sphere = {
        .origin = {4.0, 1.0, 0.0},
        .radius = 1.0,
        .material = metal_material
    };

    std::vector<Sphere> spheres;
    spheres.push_back(ground);
    spheres.push_back(matte_sphere);
    spheres.push_back(glass_sphere);
    spheres.push_back(metal_sphere);

    // random spheres
    for (int a = -11; a < 11; a++) {
        for (int b = -11; b < 11; b++) {
            float mat = random_float_in_range(0.0, 1.0);
            float temp = (float)a + 0.9*random_float_in_range(0.0, 1.0);
            float tempk = (float)b + 0.9*random_float_in_range(0.0, 1.0);
            float3 center = {temp, 0.2, tempk}; 
            float3 k = {4.0, 0.2, 0.0};
            if (length(center - k) > 0.9) {
                Material material;
                if (mat < 0.8) {
                    float3 color = {random_float_in_range(0.0,1.0), random_float_in_range(0.0, 1.0), random_float_in_range(0.0, 1.0)};
                    color = color * color;
                    material = Material {
                        .mat_ty = Matte,
                        .color = color,
                        .fuzz = 0.0,
                        .refraction_index = 0.0
                    };
                } else if (mat < 0.95) {
                    float3 color = {random_float_in_range(0.0, 0.5),random_float_in_range(0.0, 0.5),random_float_in_range(0.0, 0.5)};
                    float fuzz = random_float_in_range(0.0,0.5);
                    material = Material {
                        .mat_ty = Metal,
                        .color = color,
                        .fuzz = fuzz,
                        .refraction_index = 0.0
                    };
                } else {
                    material = Material {
                        .mat_ty = Dialectric,
                        .color = {0.0, 0.0, 0.0},
                        .fuzz = 0.0,
                        .refraction_index = 1.5
                    };
                }
                spheres.push_back(Sphere {
                    .origin = center,
                    .radius = 0.2,
                    .material = material
                });
            }
        }
    }


    // Setup driver buffer
    printf("Setting up device buffers and initializing rand state...\n");
    Sphere *d_spheres;
    int sphere_buffer_size = spheres.size() * sizeof(Sphere);

    cudaMalloc(&d_spheres, sphere_buffer_size);
    cudaMemcpy(d_spheres, spheres.data(), sphere_buffer_size, cudaMemcpyHostToDevice);


    // run trace
    int deviceID = 0; // Use device 0 (change this if you have multiple GPUs)
    int max_thread_size;

    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, deviceID);

    // Query the maximum number of threads per block
    cudaDeviceGetAttribute(&max_thread_size, cudaDevAttrMaxThreadsPerBlock, deviceID);

    // thread block params
    int tx = 8;
    int ty = 8;
    dim3 blocks(params.image_width/tx+1, params.image_height/ty+1);
    dim3 threads(tx, ty);

    int max_x = params.image_width;
    int max_y = params.image_height;
    setup_kernel<<<blocks, threads>>>(max_x, max_y, rand_states, seed);
    printf("Starting Tracing\n");
    auto start = std::chrono::high_resolution_clock::now();
    ray_trace<<<blocks, threads>>>(d_spheres, spheres.size(), d_pixels, max_x, max_y, c, params.image_height, params.image_width, rand_states);
    printf("Waiting for tracing to finish...\n");
    cudaError_t er = cudaGetLastError();

    if (er != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(er) << std::endl;
        return 1;
    }

    // copy pixels back to host
    float3 *pixels = (float3 *)malloc(pixel_count * sizeof(float3));
    cudaMemcpy(pixels, d_pixels, pixel_count * sizeof(float3), cudaMemcpyDeviceToHost);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float> elapsed = end - start;
    float seconds = elapsed.count();
    printf("Elapsed: %f\n", seconds);
    printf("Finished! Writing output to file.\n");

    // write to file
    std::ofstream fileout;
    fileout.open("image.ppm");
    fileout << "P3\n" << params.image_width << " " << params.image_height << "\n255\n";
    for (int i = 0; i < pixel_count; i++) {
        fileout << (int)pixels[i].x << " " << (int)pixels[i].y << " " << (int)pixels[i].z << "\n";
    }
    fileout.close();
    printf("Done writing to file!\n");


    
    return 0;
}
