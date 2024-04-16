//example of some shaders compiled
flat basic.vs flat.fs
texture basic.vs texture.fs
light basic.vs light.fs
skybox basic.vs skybox.fs
depth quad.vs depth.fs
multi basic.vs multi.fs

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

\light.fs

#version 330 core

in vec3 v_position;
in vec3 v_world_position;
in vec3 v_normal;
in vec2 v_uv;
in vec4 v_color;

const int MAX_LIGHTS = 4;
uniform vec3 u_light_pos[MAX_LIGHTS];
uniform vec3 u_light_color[MAX_LIGHTS];
uniform int u_num_lights;

uniform mat4 u_model;
uniform vec4 u_color;
uniform sampler2D u_texture;
uniform sampler2D u_normal_texture;
uniform sampler2D u_emissive_texture;
uniform sampler2D u_occlusion_texture;
uniform float u_time;
uniform float u_alpha_cutoff;
uniform vec3 u_ambient_light;
uniform vec3 u_emissive_light;
uniform vec3 u_light_position;
uniform vec3 u_light_color_multi;
uniform vec3 u_light_front;
uniform vec3 eye;
uniform float u_specular;
uniform float u_light_max_distance;
uniform int u_light_type;
uniform float alpha;

uniform int occlusion_option;
uniform int normal_option;
uniform int emissive_option;
uniform int single_pass_option;

#define POINTLIGHT 1
#define SPOTLIGHT 2
#define DIRECTIONALLIGHT 3

out vec4 FragColor;

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
vec3 perturbNormal(vec3 N, vec3 WP, vec2 uv, vec3 normal_pixel)
{
	normal_pixel = normal_pixel * 255./127. - 128./127.;
	mat3 TBN = cotangent_frame(N, WP, uv);
	return normalize(TBN * normal_pixel);
}


void main()
{
	vec2 uv = v_uv;
	vec4 color = u_color;
	color *= texture( u_texture, v_uv );

	vec3 N;
	vec4 final_color = vec4(1.0f);

	if (normal_option != 1) N = normalize( v_normal );

	if (single_pass_option == 0)
	{
		if (normal_option == 1)
		{
			vec3 N = texture2D(u_normal_texture, v_uv).xyz;
			N = (vec4(N * 2.0 - vec3(1.0), 0.0)*u_model).xyz;
		}

		if(color.a < u_alpha_cutoff)
			discard;

		vec3 light = u_ambient_light;

		vec3 L = u_light_position- v_world_position;
		float dist = length(L);
		L = L/dist;

		vec3 V = normalize(eye-v_world_position);
		vec3 R = normalize(reflect(-L, N));

		if (u_light_type == DIRECTIONALLIGHT)
		{
			L = u_light_front;
			float NdotL = clamp(dot(N, L), 0.0, 1.0);
			
			vec3 factor = vec3(1.0);

			if (u_specular > 0.0f)
			{
				factor *= u_specular*(clamp(pow(dot(R, V), alpha), 0.0, 1.0));
			}
			if (occlusion_option == 1){
				factor *= texture( u_occlusion_texture, v_uv ).xyz;
			}

			light += NdotL*u_light_color_multi*factor;
		}
		else if (u_light_type == POINTLIGHT || u_light_type == SPOTLIGHT)
		{
			//store the amount of diffuse light
			float NdotL = clamp(dot(N, L), 0.0, 1.0);
			
			//Compute attenuation
			float att_factor = u_light_max_distance - dist;
			att_factor /= u_light_max_distance;
			att_factor = max(att_factor, 0.0);

			vec3 factor = vec3(1.0);

			factor *= att_factor;

			if (u_specular > 0.0f)
			{
				//add specular lighting to the factor
				factor *=  u_specular*(clamp(pow(dot(R, V), alpha), 0.0, 1.0));;
			}
			if (occlusion_option == 1){
				//add occlusion lighting to the factor
				factor *= texture( u_occlusion_texture, v_uv ).xyz;
			}

			light += NdotL*u_light_color_multi * att_factor;
		}

		vec3 emissive_light = vec3(0);

		if (emissive_option == 1) {
			emissive_light = u_emissive_light*texture2D(u_emissive_texture, v_uv).xyz;
		}

		final_color.xyz = color.xyz*light + emissive_light;
		final_color.a = color.a;
	}
	else {
		vec3 light = vec3(0.0);
		for( int i = 0; i < MAX_LIGHTS; ++i )
		{
			if(i < u_num_lights)
			{
				vec3 L = normalize( u_light_pos[i] - v_world_position );
				float NdotL = max( dot(L,N), 0.0 );
				light += NdotL * u_light_color[i];
			}
		}

		final_color.xyz *= light;

	}

	FragColor = final_color;
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