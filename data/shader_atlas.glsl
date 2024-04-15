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

uniform mat4 u_model;
uniform vec4 u_color;
uniform sampler2D u_texture;
uniform sampler2D u_normal_texture;
uniform float u_time;
uniform float u_alpha_cutoff;
uniform vec3 u_ambient_light;
uniform vec3 u_emissive_light;
uniform vec3 u_light_position;
uniform vec3 u_light_color;
uniform vec3 u_light_front;
uniform vec3 eye;
uniform float u_specular;
uniform float u_light_max_distance;
uniform int u_light_type;
uniform float alpha;

uniform int normal_option;

#define POINTLIGHT 1
#define SPOTLIGHT 2
#define DIRECTIONALLIGHT 3

out vec4 FragColor;

void main()
{
	vec2 uv = v_uv;
	vec4 color = u_color;
	color *= texture( u_texture, v_uv );

	vec3 N;

	if (normal_option == 1)
	{
		vec3 N = texture2D(u_normal_texture, v_uv).xyz;
    	N = (vec4(N * 2.0 - vec3(1.0), 0.0)*u_model).xyz;
	}

	if(color.a < u_alpha_cutoff)
		discard;

	vec3 light = u_ambient_light;

	if (normal_option != 1) N = normalize( v_normal );


	vec3 L = u_light_position- v_world_position;
	float dist = length(L);
	L = L/dist;

	vec3 V = normalize(eye-v_world_position);
    vec3 R = normalize(reflect(-L, N));

	if (u_light_type == DIRECTIONALLIGHT)
	{
		L = u_light_front;
		float NdotL = clamp(dot(N, L), 0.0, 1.0);
		
		//store the amount of diffuse light
		light += NdotL*u_light_color;
	}
	else if (u_light_type == POINTLIGHT || u_light_type == SPOTLIGHT)
	{
		//store the amount of diffuse light
		float NdotL = clamp(dot(N, L), 0.0, 1.0);
		
		//Compute attenuation
		float att_factor = u_light_max_distance - dist;
		att_factor /= u_light_max_distance;
		att_factor = max(att_factor, 0.0);

        float specular = u_specular*(clamp(pow(dot(R, V), alpha), 0.0, 1.0));

		light += NdotL*u_light_color * att_factor +  NdotL*u_light_color* specular;
	}


	vec4 final_color;
	final_color.xyz = color.xyz*light + u_emissive_light;
	final_color.a = color.a;

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