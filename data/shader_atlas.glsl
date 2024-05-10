//example of some shaders compiled
flat basic.vs flat.fs
texture basic.vs texture.fs
light basic.vs light.fs
alllights basic.vs alllights.fs
skybox basic.vs skybox.fs
depth quad.vs depth.fs
multi basic.vs multi.fs
gbuffers basic.vs gbuffers.fs
deferred_global quad.vs deferred_global.fs
deferred_ws basic.vs deferred_global.fs

\basic.vs

#version 330 core

in vec3 a_vertex;
in vec3 a_normal;
in vec2 a_coord;
in vec4 a_color;

uniform vec3 u_camera_position;

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
	gl_Position = u_viewprojection * vec4( v_world_position, 1.0);
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

float computeShadow_multi(vec3 wp)
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
			shadow_factor *= computeShadow_multi(world_position);
	}
	else if (u_light_type == SPOTLIGHT  || u_light_type == POINTLIGHT)
	{	//emitted from single point in all directions

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
				shadow_factor *= computeShadow_multi(world_position);

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
		specular = factor*(clamp(pow(dot(R, V), u_alpha), 0.0, 1.0))* NdotL * u_light_color_multi * color.xyz ;//*u_specular
	}

	light += NdotL*u_light_color_multi * factor * shadow_factor + specular;

	return light;
}

\light.fs

#version 330 core

in vec3 v_position;
in vec3 v_world_position;
in vec3 v_normal;
in vec2 v_uv;
in vec4 v_color;


uniform vec4 u_color;

uniform sampler2D u_texture;
uniform sampler2D u_normalmap_texture;
uniform sampler2D u_emissive_texture;
uniform sampler2D u_metallic_roughness_texture;

uniform float u_alpha_cutoff;
uniform vec3 u_ambient_light;
uniform vec3 u_emissivef;
uniform vec3 u_light_position;
uniform vec3 u_light_color_multi;
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
		vec3 normal_pixel = texture( u_normalmap_texture, uv ).xyz; 
		N = normalize(perturbNormal( N, v_world_position, normal_pixel,v_uv));
	}
	
	//add ambient occlusion if option activated
	if (occlusion_option == 1)
		light *= texture(u_metallic_roughness_texture,uv).x;

	light = multipass(N, light, color, v_world_position);
	
	vec3 total_emitted = texture(u_emissive_texture, v_uv).xyz * u_emissivef;

	//calculate final colours
	vec4 final_color;
	final_color.xyz = color.xyz*light + total_emitted.xyz;

	final_color.a = color.a;

	FragColor = final_color;
}
   
\singlepass_functions

float computeShadow_single(vec3 wp, int pos)
{
	//project our 3D position to the shadowmap
	mat4 m = u_light_shadowmap_viewprojections[pos];

	vec4 proj_pos = u_light_shadowmap_viewprojections[pos] * vec4(wp,1.0);

	//from homogeneus space to clip space
	vec2 shadow_uv = proj_pos.xy / proj_pos.w;

	//from clip space to uv space
	shadow_uv = shadow_uv * 0.5 + vec2(0.5);

	if (shadow_uv.x < 0.0 || shadow_uv.x > 1.0 ||
		shadow_uv.y < 0.0 || shadow_uv.y > 1.0)
		return 1.0;
	//get point depth [-1 .. +1] in non-linear space
	float real_depth = (proj_pos.z - u_shadowmap_biases[pos]) / proj_pos.w;

	//normalize from [-1..+1] to [0..+1] still non-linear
	real_depth = real_depth * 0.5 + 0.5;

	//read depth from depth buffer in [0..+1] non-linear
	float shadow_depth = texture( u_shadow_textures[pos], shadow_uv).x;

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

			if (u_light_types[i] == DIRECTIONALLIGHT)
			{	//all rays are parallel, so using light front, and no attenuation
				L = u_light_fronts[i];
				NdotL = max( dot(L,N), 0.0 );
				if ( u_light_cast_shadows_arr[i] == 1.0)
					shadow_factor *= computeShadow_single(v_world_position, i);
			}
			else if (u_light_types[i] == POINTLIGHT || u_light_types[i] == SPOTLIGHT)
			{	//emitted from single point in all directions
				//vector from point to light
				L = u_light_pos[i]- v_world_position;
				float dist = length(L);
				//ignore light distance
				L = L/dist;

				NdotL = max( dot(L,N), 0.0 );

				//calculate area affected by spotlight
				if (u_light_types[i] == SPOTLIGHT)
				{
					if ( u_light_cast_shadows_arr[i] == 1.0)
						shadow_factor *= computeShadow_single(v_world_position, i);

					float cos_angle = dot( u_light_fronts[i].xyz, L );
					
					if ( cos_angle < u_light_cones_info[i].x )
						NdotL = 0.0f;
					else if ( cos_angle < u_light_cones_info[i].y )
						NdotL *= ( cos_angle - u_light_cones_info[i].x ) / ( u_light_cones_info[i].y - u_light_cones_info[i].x );
				}

				//compute attenuation factor
				float att_factor = u_light_max_distances[i] - dist;
				att_factor /= u_light_max_distances[i];
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

\alllights.fs

#version 330 core

in vec3 v_position;
in vec3 v_world_position;
in vec3 v_normal;
in vec2 v_uv;
in vec4 v_color;

const int MAX_LIGHTS = 4;
uniform vec3 u_light_pos[MAX_LIGHTS];
uniform vec3 u_light_color[MAX_LIGHTS];
uniform int u_light_types[MAX_LIGHTS];
uniform float u_light_max_distances[MAX_LIGHTS];
uniform vec2 u_light_cones_info[MAX_LIGHTS];
uniform vec3 u_light_fronts[MAX_LIGHTS];
uniform int u_num_lights;

uniform vec4 u_color;
uniform sampler2D u_texture;
uniform sampler2D u_normalmap_texture;
uniform sampler2D u_emissive_texture;
uniform sampler2D u_metallic_roughness_texture;

uniform float u_alpha_cutoff;
uniform vec3 u_ambient_light;
uniform vec3 u_emissivef;
uniform vec3 u_camera_position;
uniform float u_specular;
uniform float u_alpha;

uniform int occlusion_option;
uniform int normal_option;
uniform int single_pass_option;
uniform int specular_option;

uniform sampler2D u_shadow_textures[MAX_LIGHTS];
uniform int u_light_cast_shadows_arr[MAX_LIGHTS];
uniform mat4 u_light_shadowmap_viewprojections[MAX_LIGHTS];
uniform float u_shadowmap_biases[MAX_LIGHTS];

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
		vec3 normal_pixel = texture( u_normalmap_texture, uv ).xyz; 
		N = normalize(perturbNormal( N, v_world_position, normal_pixel,v_uv));
	}

	
	//add ambient occlusion if option activated
	if (occlusion_option == 1)
		light *= texture(u_metallic_roughness_texture,uv).x;

	light = single_pass(N, light, color);

	//vec3 spec_diffuse_light= computeDiffuseSpecular(N,u_light_type,u_light_position, u_light_color_multi, u_light_max_distance,u_light_front,u_light_cast_shadows,u_light_cone_info.x,u_light_cone_info.y, 0);
	//light+=spec_diffuse_light;
	
	vec3 total_emitted = texture(u_emissive_texture, v_uv).xyz * u_emissivef;

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
uniform vec3 u_emissivef; 
uniform float u_metallic_factor;

layout(location = 0) out vec4 FragColor;
layout(location = 1) out vec4 NormalColor;
layout(location = 2) out vec4 ExtraColor;

void main()
{
	vec2 uv = v_uv;
	vec4 color = u_color;
	color *= texture( u_texture, uv );

	if(color.a < u_alpha_cutoff)
		discard;

	vec3 N = normalize(v_normal);

	FragColor = color;
	NormalColor = vec4(N*0.5 + vec3(0.5),u_metallic_factor);

	float oclusion_factor = texture(u_metallic_roughness_texture,uv).x;
	
	vec4 emissive = texture(u_emissive_texture, uv );
	ExtraColor = vec4(emissive.xyz*u_emissivef,oclusion_factor);
	
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



\deferred_global.fs

#version 330 core

in vec3 v_position;
in vec2 v_uv;

uniform sampler2D u_color_texture;
uniform sampler2D u_normal_texture;
uniform sampler2D u_extra_texture;
uniform sampler2D u_depth_texture;

uniform vec2 u_iRes;
uniform mat4 u_inverse_viewprojection;


uniform float u_alpha_cutoff;
uniform vec3 u_ambient_light;
uniform vec3 u_emissivef;

uniform vec3 u_light_position;
uniform vec3 u_light_color_multi;
uniform vec3 u_light_front;
uniform vec3 u_camera_pos;


uniform float u_light_max_distance;
uniform int u_light_type;
uniform vec2 u_light_cone_info;
uniform float u_alpha;

uniform int occlusion_option;
uniform int normal_option;
uniform int single_pass_option;
uniform vec3 u_camera_position;

uniform int specular_option;


uniform int u_light_cast_shadows;
uniform sampler2D u_shadowmap;
uniform mat4 u_shadowmap_viewprojection;
uniform float u_shadow_bias;



#define POINTLIGHT 1
#define SPOTLIGHT 2
#define DIRECTIONALLIGHT 3


out vec4 FragColor;


#include "multipass_functions"
#include "functions"
float u_specular;
void main()
{
	vec2 uv = gl_FragCoord.xy * u_iRes.xy;

	vec4 color = texture(u_color_texture, uv);

	vec3 N = texture(u_normal_texture, uv).xyz * 2 - vec3(1.0f);
	float depth = texture(u_depth_texture, uv).x;

	if (depth == 1.0)
		discard;

	u_specular = texture(u_normal_texture, uv).a;
	vec4 material_properties = texture(u_extra_texture, uv);

	vec4 screen_pos = vec4(uv.x*2.0f-1.0f, uv.y*2.0f-1.0f, depth*2.0f-1.0f, 1.0);
	vec4 proj_worldpos = u_inverse_viewprojection * screen_pos;
	vec3 world_position = proj_worldpos.xyz / proj_worldpos.w;

	vec3 light = u_ambient_light;
	N = normalize(N);

	vec3 total_emitted = vec3(0.0);
	if (u_light_type == DIRECTIONALLIGHT)
		total_emitted= material_properties.xyz ; 
		//add ambient occlusion if option activated
		if (occlusion_option == 1)
	 		light *= material_properties.a;

	if(u_light_type != 0)
		light += multipass(N, light, color, world_position);
	
	
	

	//calculate final colours
	vec4 final_color;
	final_color.xyz = color.xyz*light + total_emitted.xyz;

	final_color.a = color.a;

	FragColor = final_color;
}