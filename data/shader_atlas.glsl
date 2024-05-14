
//example of some shaders compiled
flat basic.vs flat.fs
texture basic.vs texture.fs
multipass_light basic.vs multipass.fs
singlepass_light basic.vs singlepass.fs
skybox basic.vs skybox.fs
depth quad.vs depth.fs
multi basic.vs multi.fs
gbuffers basic.vs gbuffers.fs
deferred_global quad.vs deferred_global.fs
deferred_ws basic.vs deferred_global.fs
ssao quad.vs ssao.fs
blur quad.vs blur.fs
gamma quad.vs gamma.fs

\basic.vs

#version 330 core

in vec3 a_vertex;
in vec3 a_normal;
in vec2 a_coord;
in vec4 a_color;

uniform vec3 u_camera_pos;

uniform mat4 u_model;
uniform mat4 u_viewprojection;

//this will store the color for the pixel shader
out vec3 v_position;
out vec3 v_world_position;
out vec3 v_normal;
out vec2 v_uv;
out vec4 v_color;

uniform float u_time;

void main()
{   
    //calcule the normal in camera space (the NormalMatrix is like ViewMatrix but without traslation)
    v_normal = (u_model * vec4( a_normal, 0.0) ).xyz;
    
    //calcule the vertex in object space
    v_position = a_vertex;
    v_world_position = (u_model * vec4( v_position, 1.0) ).xyz;
    
    //store the color in the varying var to use it from the pixel shader
    v_color = a_color;

    //store the texture coordinates
    v_uv = a_coord;

    //calcule the position of the vertex using the matrices
    gl_Position = u_viewprojection * vec4( v_world_position, 1.0 );
}

\quad.vs

#version 330 core

in vec3 a_vertex;
in vec2 a_coord;
out vec2 v_uv;

void main()
{   
    v_uv = a_coord;
    gl_Position = vec4( a_vertex, 1.0 );
}


\flat.fs

#version 330 core

uniform vec4 u_color;

out vec4 FragColor;

void main()
{
    FragColor = u_color;
}

\computeShadow

#define POINTLIGHT 1
#define SPOTLIGHT 2
#define DIRECTIONALLIGHT 3

uniform int u_light_cast_shadows;
uniform sampler2D u_shadowmap;
uniform mat4 u_shadowmap_viewprojection;
uniform float u_shadow_bias;

float computeShadow(vec3 wp)
{
    //project our 3D position to the shadowmap
    vec4 proj_pos = u_shadowmap_viewprojection * vec4(wp,1.0);

    //from homogeneus space to clip space
    vec2 shadow_uv = proj_pos.xy / proj_pos.w;

    //from clip space to uv space
    shadow_uv = shadow_uv * 0.5 + vec2(0.5);

    if (shadow_uv.x < 0.0 || shadow_uv.x > 1.0 ||
    shadow_uv.y < 0.0 || shadow_uv.y > 1.0)
    {
        if (u_light_type == DIRECTIONALLIGHT) 
            return 1.0;
    }
    //get point depth [-1 .. +1] in non-linear space
    float real_depth = (proj_pos.z - u_shadow_bias) / proj_pos.w;

    //normalize from [-1..+1] to [0..+1] still non-linear
    real_depth = real_depth * 0.5 + 0.5;

    //read depth from depth buffer in [0..+1] non-linear
    float shadow_depth = texture(u_shadowmap, shadow_uv).x;

    //compute final shadow factor by comparing
    float shadow_factor = 1.0;

    //we can compare them, even if they are not linear
    if( shadow_depth < real_depth )
        shadow_factor = 0.0;

    return shadow_factor;
}

\texture.fs

#version 330 core

in vec3 v_position;
in vec3 v_world_position;
in vec3 v_normal;
in vec2 v_uv;
in vec4 v_color;

uniform vec4 u_color;
uniform sampler2D u_texture;
uniform float u_time;
uniform float u_alpha_cutoff;

out vec4 FragColor;

void main()
{
    vec2 uv = v_uv;
    vec4 color = u_color;
    color *= texture( u_texture, v_uv );

    if(color.a < u_alpha_cutoff)
        discard;

    FragColor = color;
}
\functions
mat3 cotangent_frame(vec3 N, vec3 p, vec2 uv)
{
    // get edge vectors of the pixel triangle
    vec3 dp1 = dFdx( p );
    vec3 dp2 = dFdy( p );
    vec2 duv1 = dFdx( uv );
    vec2 duv2 = dFdy( uv );
    
    // solve the linear system
    vec3 dp2perp = cross( dp2, N );
    vec3 dp1perp = cross( N, dp1 );
    vec3 T = dp2perp * duv1.x + dp1perp * duv2.x;
    vec3 B = dp2perp * duv1.y + dp1perp * duv2.y;
 
    // construct a scale-invariant frame 
    float invmax = inversesqrt( max( dot(T,T), dot(B,B) ) );
    return mat3( T * invmax, B * invmax, N );
}

// assume N, the interpolated vertex normal and 
// WP the world position
//vec3 normal_pixel = texture2D( normalmap, uv ).xyz; 
vec3 perturbNormal(vec3 N, vec3 WP, vec3 normal_pixel, vec2 uv)
{
    normal_pixel = normal_pixel * 255./127. - 128./127.;
    mat3 TBN = cotangent_frame(N, WP, uv);
    return normalize(TBN * normal_pixel);
}

\multipass_functions
float computeShadow(vec3 wp)
{
    //project our 3D position to the shadowmap
    vec4 proj_pos = u_shadowmap_viewprojection * vec4(wp,1.0);

    //from homogeneus space to clip space
    vec2 shadow_uv = proj_pos.xy / proj_pos.w;

    //from clip space to uv space
    shadow_uv = shadow_uv * 0.5 + vec2(0.5);

    if (shadow_uv.x < 0.0 || shadow_uv.x > 1.0 ||
    shadow_uv.y < 0.0 || shadow_uv.y > 1.0)
    {
        if (u_light_type == DIRECTIONALLIGHT) 
            return 1.0;
    }
    //get point depth [-1 .. +1] in non-linear space
    float real_depth = (proj_pos.z - u_shadow_bias) / proj_pos.w;

    //normalize from [-1..+1] to [0..+1] still non-linear
    real_depth = real_depth * 0.5 + 0.5;

    //read depth from depth buffer in [0..+1] non-linear
    float shadow_depth = texture(u_shadowmap, shadow_uv).x;

    //compute final shadow factor by comparing
    float shadow_factor = 1.0;

    //we can compare them, even if they are not linear
    if( shadow_depth < real_depth )
        shadow_factor = 0.0;

    return shadow_factor;
}

vec3 multipass(vec3 N, vec3 light, vec4 color, vec3 world_position)
{
    //initialize further used variables
    vec3 L;
    vec3 factor = vec3(1.0f);
    float NdotL;
    float shadow_factor = 1.0f;

    if (u_light_type == DIRECTIONALLIGHT)
    {
        //all rays are parallel, so using light front, and no attenuation
        L = u_light_front;
        NdotL = clamp(dot(N, L), 0.0, 1.0);

        if ( u_light_cast_shadows == 1.0)
            shadow_factor *= computeShadow(world_position);
    }
    else if (u_light_type == SPOTLIGHT  || u_light_type == POINTLIGHT)
    {   //emitted from single point in all directions

        //vector from point to light
        L = u_light_position - world_position;
        float dist = length(L);
        //ignore light distance
        L = L/dist;

        NdotL = clamp(dot(N, L), 0.0, 1.0);

        //calculate area affected by spotlight
        if (u_light_type == SPOTLIGHT)
        {
            if ( u_light_cast_shadows == 1.0)
                shadow_factor *= computeShadow(world_position);

            float cos_angle = dot( u_light_front.xyz, L );
            
            if ( cos_angle < u_light_cone_info.x )
                NdotL = 0.0f;
            else if ( cos_angle < u_light_cone_info.y )
                NdotL *= ( cos_angle - u_light_cone_info.x ) / ( u_light_cone_info.y - u_light_cone_info.x );
        }

        //Compute attenuation
        float att_factor = u_light_max_distance - dist;
        att_factor /= u_light_max_distance;
        att_factor = max(att_factor, 0.0);

        //accumulate light attributes in single factor
        factor *= att_factor;
    }
    
    //compute specular light if option activated, otherwise simply sum 0
    vec3 specular = vec3(0);
    if (specular_option == 1)
    {
        //view vector, from point being shaded on surface to camera (eye) 
        vec3 V = normalize(u_camera_position-world_position);
        //reflected light vector from L, hence the -L
        vec3 R = normalize(reflect(-L, N));
        //pow(dot(R, V), alpha) computes specular power
        specular = factor*u_specular*(clamp(pow(dot(R, V), u_alpha), 0.0, 1.0))* NdotL * u_light_color * color.xyz ;
    }

    light += NdotL*u_light_color * factor * shadow_factor + specular;

    return light;
}

\multipass.fs

#version 330 core

in vec3 v_position;
in vec3 v_world_position;
in vec3 v_normal;
in vec2 v_uv;
in vec4 v_color;


uniform vec4 u_color;

uniform sampler2D u_texture;
uniform sampler2D u_normal_texture;
uniform sampler2D u_emissive_texture;
uniform sampler2D u_metallic_roughness_texture;

uniform float u_alpha_cutoff;
uniform vec3 u_ambient_light;
uniform vec3 u_emissive_factor;
uniform vec3 u_light_position;
uniform vec3 u_light_color;
uniform vec3 u_light_front;
uniform vec3 u_camera_position;
uniform float u_specular;
uniform float u_light_max_distance;
uniform int u_light_type;
uniform vec2 u_light_cone_info;
uniform float u_alpha;

uniform int occlusion_option;
uniform int normal_option;
uniform int single_pass_option;
uniform int specular_option;
uniform int emissive_option;

uniform int u_light_cast_shadows;
uniform sampler2D u_shadowmap;
uniform mat4 u_shadowmap_viewprojection;
uniform float u_shadow_bias;


#define POINTLIGHT 1
#define SPOTLIGHT 2
#define DIRECTIONALLIGHT 3

out vec4 FragColor;

#include "functions"
#include "multipass_functions"
    
void main()
{
    vec2 uv = v_uv;
    vec4 color = u_color;
    color *= texture( u_texture, uv );

    if(color.a < u_alpha_cutoff)
        discard;

    vec3 N = normalize( v_normal );
    vec3 light = u_ambient_light;

    //calculate normal with normalmap if option activated
    if(normal_option == 1){
        vec3 normal_pixel = texture( u_normal_texture, uv ).xyz; 
        N = normalize(perturbNormal( N, v_world_position, normal_pixel,v_uv));
    }
    
    //add ambient occlusion if option activated
    if (occlusion_option == 1)
        light *= texture(u_metallic_roughness_texture,uv).x;

    light = multipass(N, light, color, v_world_position);
    
    vec3 total_emitted = vec3(0.0f);
    
    if (emissive_option == 1) 
        total_emitted = texture(u_emissive_texture, v_uv).xyz * u_emissive_factor;

    //calculate final colours
    vec4 final_color;
    final_color.xyz = color.xyz*light + total_emitted.xyz;

    final_color.a = color.a;

    FragColor = final_color;
}
   
\singlepass_functions

float computeShadow(vec3 wp, int pos)
{
    //project our 3D position to the shadowmap
    mat4 m = u_light_shadowmap_viewprojection[pos];

    vec4 proj_pos = u_light_shadowmap_viewprojection[pos] * vec4(wp,1.0);

    //from homogeneus space to clip space
    vec2 shadow_uv = proj_pos.xy / proj_pos.w;

    //from clip space to uv space
    shadow_uv = shadow_uv * 0.5 + vec2(0.5);

    if (shadow_uv.x < 0.0 || shadow_uv.x > 1.0 ||
        shadow_uv.y < 0.0 || shadow_uv.y > 1.0)
        return 1.0;
    //get point depth [-1 .. +1] in non-linear space
    float real_depth = (proj_pos.z - u_shadowmap_bias[pos]) / proj_pos.w;

    //normalize from [-1..+1] to [0..+1] still non-linear
    real_depth = real_depth * 0.5 + 0.5;

    //read depth from depth buffer in [0..+1] non-linear
    float shadow_depth = texture( u_shadow_texture[pos], shadow_uv).x;

    //compute final shadow factor by comparing
    float shadow_factor = 1.0;

    //we can compare them, even if they are not linear
    if( shadow_depth < real_depth )
        shadow_factor = 0.0;

    return shadow_factor;
}

vec3 single_pass(vec3 N, vec3 light, vec4 color)
{
    for( int i = 0; i < MAX_LIGHTS; ++i )
    {
        if(i < u_num_lights)
        {
            //initialize further used variables
            vec3 factor = vec3(1.0);
            vec3 L;
            float shadow_factor = 1.0f;
            float NdotL;

            if (u_light_type[i] == DIRECTIONALLIGHT)
            {   //all rays are parallel, so using light front, and no attenuation
                L = u_light_front[i];
                NdotL = max( dot(L,N), 0.0 );
                if ( u_light_cast_shadows[i] == 1.0)
                    shadow_factor *= computeShadow(v_world_position, i);
            }
            else if (u_light_type[i] == POINTLIGHT || u_light_type[i] == SPOTLIGHT)
            {   //emitted from single point in all directions
                //vector from point to light
                L = u_light_pos[i]- v_world_position;
                float dist = length(L);
                //ignore light distance
                L = L/dist;

                NdotL = max( dot(L,N), 0.0 );

                //calculate area affected by spotlight
                if (u_light_type[i] == SPOTLIGHT)
                {
                    if ( u_light_cast_shadows[i] == 1.0)
                        shadow_factor *= computeShadow(v_world_position, i);

                    float cos_angle = dot( u_light_front[i].xyz, L );
                    
                    if ( cos_angle < u_light_cone_info[i].x )
                        NdotL = 0.0f;
                    else if ( cos_angle < u_light_cone_info[i].y )
                        NdotL *= ( cos_angle - u_light_cone_info[i].x ) / ( u_light_cone_info[i].y - u_light_cone_info[i].x );
                }

                //compute attenuation factor
                float att_factor = u_light_max_distance[i] - dist;
                att_factor /= u_light_max_distance[i];
                att_factor = max(att_factor, 0.0);

                //accumulate into single factor
                factor *= att_factor;
            }

            //compute specular light if option activated, otherwise simply sum 0
            vec3 specular = vec3(0);
            if (specular_option == 1)
            {
                vec3 V = normalize(u_camera_position-v_world_position);
                vec3 R = normalize(reflect(-L, N));
                specular = factor*u_specular*(clamp(pow(dot(R, V), u_alpha), 0.0, 1.0)) * NdotL * u_light_color[i] * color.xyz;
            }

            //accumulate computed light into final light
            light += NdotL*u_light_color[i]*factor*shadow_factor + specular;
        }
    }

    return light;
}

\singlepass.fs

#version 330 core

in vec3 v_position;
in vec3 v_world_position;
in vec3 v_normal;
in vec2 v_uv;
in vec4 v_color;

const int MAX_LIGHTS = 4;
uniform vec3 u_light_pos[MAX_LIGHTS];
uniform vec3 u_light_color[MAX_LIGHTS];
uniform int u_light_type[MAX_LIGHTS];
uniform float u_light_max_distance[MAX_LIGHTS];
uniform vec2 u_light_cone_info[MAX_LIGHTS];
uniform vec3 u_light_front[MAX_LIGHTS];
uniform int u_num_lights;

uniform vec4 u_color;
uniform sampler2D u_texture;
uniform sampler2D u_normal_texture;
uniform sampler2D u_emissive_texture;
uniform sampler2D u_metallic_roughness_texture;

uniform float u_alpha_cutoff;
uniform vec3 u_ambient_light;
uniform vec3 u_emissive_factor;
uniform vec3 u_camera_position;
uniform float u_specular;
uniform float u_alpha;

uniform int occlusion_option;
uniform int normal_option;
uniform int single_pass_option;
uniform int specular_option;
uniform int emissive_option;

uniform sampler2D u_shadow_texture[MAX_LIGHTS];
uniform int u_light_cast_shadows[MAX_LIGHTS];
uniform mat4 u_light_shadowmap_viewprojection[MAX_LIGHTS];
uniform float u_shadowmap_bias[MAX_LIGHTS];

#define POINTLIGHT 1
#define SPOTLIGHT 2
#define DIRECTIONALLIGHT 3

out vec4 FragColor;

#include "functions"
#include "singlepass_functions"

void main()
{
    vec2 uv = v_uv;
    vec4 color = u_color;
    color *= texture( u_texture, uv );

    if(color.a < u_alpha_cutoff)
        discard;

    vec3 N = normalize( v_normal );
    vec3 light = u_ambient_light;

    //calculate normal with normalmap if option activated
    if(normal_option == 1){
        vec3 normal_pixel = texture( u_normal_texture, uv ).xyz; 
        N = normalize(perturbNormal( N, v_world_position, normal_pixel,v_uv));
    }

    
    //add ambient occlusion if option activated
    if (occlusion_option == 1)
        light *= texture(u_metallic_roughness_texture,uv).x;

    light = single_pass(N, light, color);
    
    vec3 total_emitted = vec3(0.0f);
    if (emissive_option == 1)
        total_emitted = texture(u_emissive_texture, v_uv).xyz * u_emissive_factor;

    //calculate final colours
    vec4 final_color;
    final_color.xyz = color.xyz*light + total_emitted.xyz;

    final_color.a = color.a;

    FragColor = final_color;
}

\gbuffers.fs

#version 330 core

in vec3 v_position;
in vec3 v_world_position;
in vec3 v_normal;
in vec2 v_uv;

uniform vec4 u_color;
uniform sampler2D u_texture;
uniform float u_time;
uniform float u_alpha_cutoff;

//added
uniform sampler2D u_metallic_roughness_texture;
uniform sampler2D u_emissive_texture;
uniform sampler2D u_normal_texture;

uniform vec3 u_emissive_factor;
uniform float u_metallic_factor;
uniform float u_metallic_roughness;

uniform int normal_option;
uniform int emissive_option;
uniform int occlusion_option;

layout(location = 0) out vec4 FragColor;
layout(location = 1) out vec4 NormalColor;
layout(location = 2) out vec4 ExtraColor;
layout(location = 3) out vec4 MetalnessColor;

#include "functions"

void main()
{
    vec2 uv = v_uv;
    vec4 color = u_color;
    color *= texture( u_texture, uv );

    if(color.a < u_alpha_cutoff)
        discard;

    vec3 N = normalize(v_normal);
    if(normal_option == 1){
        vec3 normal_pixel = texture( u_normal_texture, uv ).xyz;
        N = perturbNormal(N,v_world_position, normal_pixel, uv);
    }

    FragColor = color;
    NormalColor = vec4(N*0.5 + vec3(0.5),1.0);
    NormalColor.a = u_metallic_factor;

    vec4 material_properties = texture(u_metallic_roughness_texture, uv );
    material_properties.y = pow(material_properties.y, material_properties.x);
    material_properties.z = pow(material_properties.z, material_properties.y);
    vec3 emissive = (emissive_option == 1) ? texture(u_emissive_texture, uv ).xyz*u_emissive_factor : vec3(0);
    ExtraColor.xyz = emissive;
    ExtraColor.a = (occlusion_option == 1) ? material_properties.x : 1;
    MetalnessColor = vec4(u_metallic_roughness, u_metallic_factor, 0, 0);
}

\skybox.fs

#version 330 core

in vec3 v_position;
in vec3 v_world_position;

uniform samplerCube u_texture;
uniform vec3 u_camera_position;
out vec4 FragColor;

void main()
{
    vec3 E = v_world_position - u_camera_position;
    vec4 color = texture( u_texture, E );
    FragColor = color;
}


\multi.fs

#version 330 core

in vec3 v_position;
in vec3 v_world_position;
in vec3 v_normal;
in vec2 v_uv;

uniform vec4 u_color;
uniform sampler2D u_texture;
uniform float u_time;
uniform float u_alpha_cutoff;

layout(location = 0) out vec4 FragColor;
layout(location = 1) out vec4 NormalColor;

void main()
{
    vec2 uv = v_uv;
    vec4 color = u_color;
    color *= texture( u_texture, uv );

    if(color.a < u_alpha_cutoff)
        discard;

    vec3 N = normalize(v_normal);

    FragColor = color;
    NormalColor = vec4(N,1.0);
}


\depth.fs

#version 330 core

uniform vec2 u_camera_nearfar;
uniform sampler2D u_texture; //depth map
in vec2 v_uv;
out vec4 FragColor;

void main()
{
    float n = u_camera_nearfar.x;
    float f = u_camera_nearfar.y;
    float z = texture2D(u_texture,v_uv).x;
    if( n == 0.0 && f == 1.0 )
        FragColor = vec4(z);
    else
        FragColor = vec4( n * (z + 1.0) / (f + n - z * (f - n)) );
}


\instanced.vs

#version 330 core

in vec3 a_vertex;
in vec3 a_normal;
in vec2 a_coord;

in mat4 u_model;

uniform vec3 u_camera_pos;

uniform mat4 u_viewprojection;

//this will store the color for the pixel shader
out vec3 v_position;
out vec3 v_world_position;
out vec3 v_normal;
out vec2 v_uv;

void main()
{   
    //calcule the normal in camera space (the NormalMatrix is like ViewMatrix but without traslation)
    v_normal = (u_model * vec4( a_normal, 0.0) ).xyz;
    
    //calcule the vertex in object space
    v_position = a_vertex;
    v_world_position = (u_model * vec4( a_vertex, 1.0) ).xyz;
    
    //store the texture coordinates
    v_uv = a_coord;

    //calcule the position of the vertex using the matrices
    gl_Position = u_viewprojection * vec4( v_world_position, 1.0 );
}

\deferredpass_functions

float computeShadow(vec3 wp)
{
    //project our 3D position to the shadowmap
    vec4 proj_pos = u_shadowmap_viewprojection * vec4(wp,1.0);

    //from homogeneus space to clip space
    vec2 shadow_uv = proj_pos.xy / proj_pos.w;

    //from clip space to uv space
    shadow_uv = shadow_uv * 0.5 + vec2(0.5);

    if (shadow_uv.x < 0.0 || shadow_uv.x > 1.0 ||
    shadow_uv.y < 0.0 || shadow_uv.y > 1.0)
    {
        if (u_light_type == DIRECTIONALLIGHT) 
            return 1.0;
    }
    //get point depth [-1 .. +1] in non-linear space
    float real_depth = (proj_pos.z - u_shadow_bias) / proj_pos.w;

    //normalize from [-1..+1] to [0..+1] still non-linear
    real_depth = real_depth * 0.5 + 0.5;

    //read depth from depth buffer in [0..+1] non-linear
    float shadow_depth = texture(u_shadowmap, shadow_uv).x;

    //compute final shadow factor by comparing
    float shadow_factor = 1.0;

    //we can compare them, even if they are not linear
    if( shadow_depth < real_depth )
        shadow_factor = 0.0;

    return shadow_factor;
}

// Geometry Term: Geometry masking/shadowing due to microfacets
float GGX(float NdotV, float k){
    return NdotV / (NdotV * (1.0 - k) + k);
}
    
float G_Smith( float NdotV, float NdotL, float roughness)
{
    float k = pow(roughness + 1.0, 2.0) / 8.0;
    return GGX(NdotL, k) * GGX(NdotV, k);
}

// Fresnel term with scalar optimization(f90=1)
float F_Schlick( const in float VoH, 
const in float f0)
{
    float f = pow(1.0 - VoH, 5.0);
    return f0 + (1.0 - f0) * f;
}

// Fresnel term with colorized fresnel
vec3 F_Schlick( const in float VoH, 
const in vec3 f0)
{
    float f = pow(1.0 - VoH, 5.0);
    return f0 + (vec3(1.0) - f0) * f;
}

// Diffuse Reflections: Disney BRDF using retro-reflections using F term, this is much more complex!!
// float Fd_Burley ( const in float NoV, const in float NoL,
// const in float LoH, 
// const in float linearRoughness)
// {
//         float f90 = 0.5 + 2.0 * linearRoughness * LoH * LoH;
//         float lightScatter = F_Schlick(NoL, 1.0, f90);
//         float viewScatter  = F_Schlick(NoV, 1.0, f90);
//         return lightScatter * viewScatter * RECIPROCAL_PI;
// }

// Normal Distribution Function using GGX Distribution
float D_GGX (   const in float NoH, 
const in float linearRoughness )
{
    float a2 = linearRoughness * linearRoughness;
    float f = (NoH * NoH) * (a2 - 1.0) + 1.0;
    return a2 / (PI * f * f);
}

//Cook Torrance specular
vec3 specularBRDF( float roughness, vec3 f0, 
float NoH, float NoV, float NoL, float LoH )
{
float a = roughness * roughness;

// Normal Distribution Function
float D = D_GGX( NoH, a );

    // Fresnel Function
    vec3 F = F_Schlick( LoH, f0 );

    // Visibility Function (shadowing/masking)
    float G = G_Smith( NoV, NoL, roughness );
        
    // Norm factor
    vec3 spec = D * G * F;
    spec /= (4.0 * NoL * NoV + 1e-6);

    return spec;
}


vec3 multipass(vec3 N, vec3 light, vec4 color, vec3 world_position)
{
    //initialize further used variables
    vec3 L;
    vec3 factor = vec3(1.0f);
    float NdotL;
    float shadow_factor = 1.0f;

    vec2 uv = (gl_FragCoord.xy * u_iRes);
    vec4 metallic_roughness = texture(u_metallic_texture, uv);

    //we compute the reflection in base to the color and the metalness
    vec3 f0 = mix( vec3(0.5), color.xyz, metallic_roughness.y );
    //metallic materials do not have diffuse
    vec3 diffuseColor = (1.0 - metallic_roughness.y) * color.xyz;

    if (u_light_type == DIRECTIONALLIGHT)
    {
        //all rays are parallel, so using light front, and no attenuation
        L = u_light_front;
        NdotL = clamp(dot(N, L), 0.0, 1.0);

        vec3 L = u_light_front;
        vec3 V = normalize(u_camera_pos-world_position);
        //reflected light vector from L, hence the -L
        vec3 R = normalize(reflect(N, V));
        vec4 cubeColor = textureCube( u_cube_texture, R );

        // color.rgb = (color.rgb*color.a) + (cubeColor.rgb * (1 - color.a));
        // color = mix(color, cubeColor, metallic_roughness.x);
        // color = cubeColor;

        if ( u_light_cast_shadows == 1.0)
            shadow_factor *= computeShadow(world_position);
    }
    else if (u_light_type == SPOTLIGHT  || u_light_type == POINTLIGHT)
    {   //emitted from single point in all directions

        //vector from point to light
        L = u_light_position - world_position;
        float dist = length(L);
        //ignore light distance
        L = L/dist;

        NdotL = clamp(dot(N, L), 0.0, 1.0);

        //calculate area affected by spotlight
        if (u_light_type == SPOTLIGHT)
        {
            if ( u_light_cast_shadows == 1.0)
                shadow_factor *= computeShadow(world_position);

            float cos_angle = dot( u_light_front.xyz, L );
            
            if ( cos_angle < u_light_cone_info.x )
                NdotL = 0.0f;
            else if ( cos_angle < u_light_cone_info.y )
                NdotL *= ( cos_angle - u_light_cone_info.x ) / ( u_light_cone_info.y - u_light_cone_info.x );
        }

        //Compute attenuation
        float att_factor = u_light_max_distance - dist;
        att_factor /= u_light_max_distance;
        att_factor = max(att_factor, 0.0);

        //accumulate light attributes in single factor
        factor *= att_factor;
    }
    
    //compute specular light if option activated, otherwise simply sum 0
    vec3 lightParams = vec3(0);
    if (specular_option == 1)
    {
        vec3 V = normalize(u_camera_pos - world_position);
        vec3 H = (V + L) / 2;
		float NdotV = dot(N, V);
		float NdotH = dot(N, H);
		float LdotH = dot(L, H);

        //compute the specular
        vec3 Fr_d = specularBRDF(metallic_roughness.x, f0, NdotH, NdotV, NdotL, LdotH);
        // Here we use the Burley, but you can replace it by the Lambert.
        // linearRoughness = squared roughness
        float linearRoughness = metallic_roughness.x*metallic_roughness.x;
        vec3 Fd_d = diffuseColor * color.xyz/PI; 

        //add diffuse and specular reflection
        vec3 direct = Fr_d + Fd_d;

        //compute how much light received the pixel
        lightParams = max(0.0f, NdotL) * u_light_color * factor * shadow_factor;// * direct;
    }

    light += NdotL*u_light_color * factor * shadow_factor + lightParams;
    light *= color.xyz;

    return light;
}

\deferred_global.fs

#version 330 core

in vec3 v_position;
in vec2 v_uv;

uniform sampler2D u_color_texture;
uniform sampler2D u_normal_texture;
uniform sampler2D u_extra_texture;
uniform sampler2D u_metallic_texture;
uniform sampler2D u_depth_texture;

uniform vec2 u_iRes;
uniform mat4 u_inverse_viewprojection;

uniform float u_alpha_cutoff;
uniform vec3 u_ambient_light;
uniform vec3 u_light_position;
uniform vec3 u_light_color;
uniform vec3 u_light_front;
uniform vec3 u_camera_pos;

uniform float u_light_max_distance;
uniform int u_light_type;
uniform vec2 u_light_cone_info;

uniform int occlusion_option;
uniform int normal_option;
uniform int specular_option;

uniform int u_light_cast_shadows;
uniform sampler2D u_shadowmap;
uniform mat4 u_shadowmap_viewprojection;
uniform float u_shadow_bias;

uniform samplerCube u_cube_texture;

uniform sampler2D u_ao_texture;
uniform int u_use_ssao;
uniform int use_degamma;

#define POINTLIGHT 1
#define SPOTLIGHT 2
#define DIRECTIONALLIGHT 3

out vec4 FragColor;

#define RECIPROCAL_PI 0.3183098861837697
#define PI 3.1415926538

#extension GL_NV_shadow_samplers_cube : enable

#include "deferredpass_functions"
vec3 degamma(vec3 c)
{
	return pow(c,vec3(2.2));
}
void main()
{
    vec2 uv = (gl_FragCoord.xy * u_iRes);

    vec4 color = texture(u_color_texture, uv);
    
    vec3 N = texture(u_normal_texture, uv).xyz * 2 - vec3(1.0f);
    float depth = texture(u_depth_texture, uv).x;

    if (use_degamma == 1)
        N = degamma(texture(u_normal_texture, uv).xyz) * 2  - vec3(1.0f);
        color = vec4(degamma(color.xyz), 1.0);

    if (depth == 1.0f)
        discard;

    vec4 screen_pos = vec4(uv.x*2.0f-1.0f, uv.y*2.0f-1.0f, depth*2.0f-1.0f, 1.0);
    vec4 proj_worldpos = u_inverse_viewprojection * screen_pos;
    vec3 world_position = proj_worldpos.xyz / proj_worldpos.w;

    vec4 material_properties = texture(u_extra_texture, uv);

    N = normalize(N);
    
    int factor = (u_light_type == DIRECTIONALLIGHT) ? 1 : 0;
    
    vec3 ambient;

    if(u_use_ssao == 1)
    {
        //read the ao_factor for this pixel
        float ao_factor = texture( u_ao_texture, uv ).x;
        //we could play with the curve to have more control
        ao_factor = pow( ao_factor, 3.0 );
        //weight the ambient light by it
        ambient= u_ambient_light * ao_factor ;

    }
    else
    {    
         
         ambient= u_ambient_light*material_properties.a;
    }
    
    vec3 light = (u_light_type == DIRECTIONALLIGHT) ? ambient : u_ambient_light;


    light = multipass(N, light, color, world_position);
    
    //calculate final colours
    vec4 final_color = vec4(0);

    final_color.a = color.a;

    vec4 cubeColor;
    
    final_color.xyz = light + factor*material_properties.xyz;

    FragColor = final_color;
}


\ssao.fs

#version 330 core

in vec3 v_position;
in vec2 v_uv;

uniform sampler2D u_normal_texture; //blur
uniform sampler2D u_depth_texture;

uniform vec2 u_iRes;
uniform mat4 u_inverse_viewprojection;
uniform mat4 u_viewprojection;
uniform float u_radius;

uniform float near;
uniform float far;


uniform vec3 u_points[64]; //puntos aletorios dentro de una esfera
out vec4 FragColor;

float depthToLinear(float z)
{
    return near * (z + 1.0) / (far + near - z * (far - near));
}


void main()
{
	vec2 uv = v_uv + u_iRes * 0.5; //gl_FragCoord.xy * u_iRes.xy;

	vec3 N = texture(u_normal_texture, uv).xyz * 2 - vec3(1.0f);
	N = normalize(N);
	float depth = texture(u_depth_texture, uv).x;

	if (depth == 1.0) //skybox
		discard;

	vec4 screen_pos = vec4(uv.x*2.0f-1.0f, uv.y*2.0f-1.0f, depth*2.0f-1.0f, 1.0);
	vec4 proj_worldpos = u_inverse_viewprojection * screen_pos;
	vec3 world_position = proj_worldpos.xyz / proj_worldpos.w;


    //lets use 64 samples
    const int samples = 64;
    int num = samples; //num samples that passed the are outside

    
    for(int i = 0; i < samples; ++i) {
        vec3 random_point = u_points[i]; 

        //check in which side of the normal
        if(dot(N, random_point) < 0.0)
            random_point *= -1.0;

        //compute is world position using the random
        vec3 p = world_position + random_point * u_radius;

        //find the uv in the depth buffer of this point
        vec4 proj = u_viewprojection * vec4(p, 1.0);
        proj.xy /= proj.w; //convert to clipspace from homogeneous

        //apply a tiny bias to its z before converting to clip-space
        proj.z = (proj.z - 0.005) / proj.w;
        proj.xyz = proj.xyz * 0.5 + vec3(0.5); //to [0..1]

        //read p true depth
        float pdepth = texture(u_depth_texture, proj.xy).x;

        //linearize the depth
        pdepth = depthToLinear(pdepth);
        float projz = depthToLinear(proj.z);

        //compare true depth with its depth
        float diff = pdepth - projz;

        if(diff < 0.0 && abs(diff) < 0.001) //if true depth smaller, is inside
            num--; //remove this point from the list of visible
    }

    //finally, compute the AO factor as the ratio of visible points
    float ao = float(num) / float(samples);

    FragColor = vec4(ao, ao, ao, 1.0);


}

\blur.fs

#version 330 core
  
in vec2 v_uv;
  
uniform sampler2D u_ssao_texture;
uniform vec2 u_iRes;

out vec4 FragColor;

void main() {
    
   vec4 blur_color = vec4(0.0);
    for (int x = -4; x < 4; ++x) 
    {
        for (int y = -4; y < 4; ++y) 
        {
            vec2 offset = vec2(float(x), float(y)) * u_iRes;
            blur_color += texture(u_ssao_texture, v_uv + offset);
        }
    }
    FragColor = blur_color / (8.0 * 8.0);
}  









\gamma.fs
#version 330 core

in vec2 v_uv;
uniform sampler2D u_texture;

out vec4 FragColor;

void main()
{
    vec4 color = texture(u_texture,v_uv);
    //color.xyz = pow(color,vec3(1.0/2.2));
    color.r = pow(color.r, 1.0/2.2);
    color.g = pow(color.g, 1.0/2.2);
    color.b = pow(color.b, 1.0/2.2);
    
    
    FragColor = color;
}





