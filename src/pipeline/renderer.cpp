#include "renderer.h"

#include <algorithm> //sort

#include "camera.h"
#include "../gfx/gfx.h"
#include "../gfx/shader.h"
#include "../gfx/mesh.h"
#include "../gfx/texture.h"
#include "../gfx/fbo.h"
#include "../pipeline/prefab.h"
#include "../pipeline/material.h"
#include "../pipeline/animation.h"
#include "../utils/utils.h"
#include "../extra/hdre.h"
#include "../core/ui.h"

#include "scene.h"

#define N_LIGHTS 4


using namespace SCN;

//some globals
GFX::Mesh sphere;

void Renderer::extractRenderables(SCN::Node* node, Camera* camera)
{
	if (!node->visible)
		return;
	
	//Compute global matrix
	Matrix44 node_model = node->getGlobalMatrix(true);

	//If node has mesh->render
	if (node->mesh && node->material)
	{
		//Compute bounding box of the object in world space by using box transformed to world space
		BoundingBox world_bounding = transformBoundingBox(node_model, node->mesh->box);

		//If bounding box inside the camera frustum then the object is probably visible
		if (camera->testBoxInFrustum(world_bounding.center, world_bounding.halfsize))
		{
			Renderable re;
			re.model = node_model;
			re.mesh  = node->mesh;
			re.material = node->material;
			re.dist_to_cam = camera->eye.distance(world_bounding.center);
			re.bounding = world_bounding;
			renderables.push_back(re);

			if (re.material->alpha_mode == eAlphaMode::BLEND)
				alpha_renderables.push_back(re);
			else
				opaque_renderables.push_back(re);
		}
	}

	//Iterate recursively with children
	for (int i = 0; i < node->children.size(); ++i)
		extractRenderables(node->children[i], camera);
}

Renderer::Renderer(const char* shader_atlas_filename)
{
	render_wireframe = false;
	render_boundaries = false;
	scene = nullptr;
	skybox_cubemap = nullptr;

	if (!GFX::Shader::LoadAtlas(shader_atlas_filename))
		exit(1);
	GFX::checkGLErrors();

	use_multipass_lights = true;
	use_no_texture = false;
	use_normal_map = false;
	use_emissive   = false;
	use_specular   = false;
	use_occlusion  = false;

	sphere.createSphere(1.0f);
	sphere.uploadToVRAM();

	shadowmap_size = 1024;
	mainLight = nullptr;
}

void Renderer::extractSceneInfo(SCN::Scene* scene, Camera* camera)
{
	renderables.clear();
	lights.clear();
	opaque_renderables.clear();
	alpha_renderables.clear();
	mainLight = nullptr;

	for (BaseEntity* ent : scene->entities)
	{
		if (!ent->visible)
			continue;
		
		//Is a prefab
		if (ent->getType() == eEntityType::PREFAB)
		{
			PrefabEntity* pent = (SCN::PrefabEntity*) ent;
			if (pent->prefab)
				extractRenderables(&pent->root, camera);
		}
		else if (ent->getType() == eEntityType::LIGHT)
		{
			LightEntity* light = (SCN::LightEntity*) ent;

			mat4 globalMatrix = light->root.getGlobalMatrix();

			if ( light->light_type == eLightType::DIRECTIONAL || camera->testSphereInFrustum(globalMatrix.getTranslation(), light->max_distance))
				lights.push_back(light);

			if (!mainLight && light->light_type == eLightType::DIRECTIONAL)
				mainLight = light;
		}
	}
}

bool Renderer::renderableComparator(const Renderable& a, const Renderable& b)
{
	return a.dist_to_cam >= b.dist_to_cam;
}

void Renderer::setupScene()
{
	if (scene->skybox_filename.size())
		skybox_cubemap = GFX::Texture::Get(std::string(scene->base_folder + "/" + scene->skybox_filename).c_str());
	else
		skybox_cubemap = nullptr;
}

void Renderer::generateShadowMaps(Camera* camera)
{
	for (int i = 0; i < lights.size(); ++i)
	{
		LightEntity* light = lights[i];

		if (!light->cast_shadows || light->light_type == eLightType::POINT)
			continue;

		if (light->shadowMapFBO == nullptr || light->shadowMapFBO->width != shadowmap_size)
		{
			if (light->shadowMapFBO)
				delete light->shadowMapFBO;
			light->shadowMapFBO = new GFX::FBO();
			light->shadowMapFBO->setDepthOnly(shadowmap_size, shadowmap_size);
		}

		Camera light_camera;
		vec3 up = vec3(0, 1, 0);

		vec3 pos = light->getGlobalPosition();
		//pos = camera->eye;
		light_camera.lookAt(pos, pos - light->getFront(), up);

		light->shadowMapFBO->bind();
		glClear(GL_DEPTH_BUFFER_BIT);

		if (light->light_type == eLightType::DIRECTIONAL)
		{
			//light = mainLight;

			float halfArea = light->area / 2.0f;
			light_camera.setOrthographic(-halfArea, halfArea, -halfArea, halfArea, light->near_distance, light->max_distance);
		}
		else if (light->light_type == eLightType::SPOT)
		{
			light_camera.setPerspective(light->cone_info.y * 2.0f, 1.0f, light->near_distance, light->max_distance);
		}

		light_camera.enable();

		light->shadowMap_viewProjection = light_camera.viewprojection_matrix;

		for (Renderable& re : opaque_renderables)
		{
			if (light_camera.testBoxInFrustum(re.bounding.center, re.bounding.halfsize))
				renderMeshWithMaterialPlain(re.model, re.mesh, re.material);
		}
		light->shadowMapFBO->unbind();
	}
}

void Renderer::renderScene(SCN::Scene* scene, Camera* camera)
{
	this->scene = scene;
	setupScene();
	extractSceneInfo(scene, camera);
	generateShadowMaps(camera);
	camera->enable();

	sort(alpha_renderables.begin(), alpha_renderables.end(), renderableComparator);

	std::vector<Renderable> sortedRenderables;

	sortedRenderables.reserve( opaque_renderables.size() + alpha_renderables.size() ); // preallocate memory
	sortedRenderables.insert( sortedRenderables.end(), opaque_renderables.begin()   , opaque_renderables.end()    );
	sortedRenderables.insert( sortedRenderables.end(), alpha_renderables.begin(), alpha_renderables.end() );

	glDisable(GL_BLEND);
	glEnable(GL_DEPTH_TEST);

	//set the clear color (the background color)
	glClearColor(scene->background_color.x, scene->background_color.y, scene->background_color.z, 1.0);

	// Clear the color and the depth buffer
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	GFX::checkGLErrors();

	//render skybox
	if(skybox_cubemap)
		renderSkybox(skybox_cubemap);

	//render entities
	for (Renderable& re : sortedRenderables)
	{
		if (camera->testBoxInFrustum(re.bounding.center, re.bounding.halfsize))
			renderMeshWithMaterialLights(re.model, re.mesh, re.material);
	}
}


void Renderer::renderSkybox(GFX::Texture* cubemap)
{
	Camera* camera = Camera::current;

	glDisable(GL_BLEND);
	glDisable(GL_DEPTH_TEST);
	glDisable(GL_CULL_FACE);
	if (render_wireframe)
		glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);

	GFX::Shader* shader = GFX::Shader::Get("skybox");
	if (!shader)
		return;
	shader->enable();

	Matrix44 m;
	m.setTranslation(camera->eye.x, camera->eye.y, camera->eye.z);
	m.scale(10, 10, 10);
	shader->setUniform("u_model", m);
	cameraToShader(camera, shader);
	shader->setUniform("u_texture", cubemap, 0);
	sphere.render(GL_TRIANGLES);
	shader->disable();
	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	glEnable(GL_DEPTH_TEST);
}

//renders a node of the prefab and its children
void Renderer::renderNode(SCN::Node* node, Camera* camera)
{
	if (!node->visible)
		return;

	//compute global matrix
	Matrix44 node_model = node->getGlobalMatrix(true);

	//does this node have a mesh? then we must render it
	if (node->mesh && node->material)
	{
		//compute the bounding box of the object in world space (by using the mesh bounding box transformed to world space)
		BoundingBox world_bounding = transformBoundingBox(node_model,node->mesh->box);
		
		//if bounding box is inside the camera frustum then the object is probably visible
		if (camera->testBoxInFrustum(world_bounding.center, world_bounding.halfsize) )
		{
			if(render_boundaries)
				node->mesh->renderBounding(node_model, true);
			renderMeshWithMaterial(node_model, node->mesh, node->material);
		}
	}

	//iterate recursively with children
	for (int i = 0; i < node->children.size(); ++i)
		renderNode( node->children[i], camera);
}

//renders a mesh given its transform and material
void Renderer::renderMeshWithMaterial(const Matrix44 model, GFX::Mesh* mesh, SCN::Material* material)
{
	//in case there is nothing to do
	if (!mesh || !mesh->getNumVertices() || !material )
		return;
    assert(glGetError() == GL_NO_ERROR);

	//define locals to simplify coding
	GFX::Shader* shader = NULL;
	GFX::Texture* texture = NULL;
	Camera* camera = Camera::current;
	
	texture = material->textures[SCN::eTextureChannel::ALBEDO].texture;
	//texture = material->emissive_texture;
	//texture = material->metallic_roughness_texture;
	//texture = material->normal_texture;
	//texture = material->occlusion_texture;
	if (texture == NULL)
		texture = GFX::Texture::getWhiteTexture(); //a 1x1 white texture

	//select the blending
	if (material->alpha_mode == SCN::eAlphaMode::BLEND)
	{
		glEnable(GL_BLEND);
		glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	}
	else
		glDisable(GL_BLEND);

	//select if render both sides of the triangles
	if(material->two_sided)
		glDisable(GL_CULL_FACE);
	else
		glEnable(GL_CULL_FACE);
    assert(glGetError() == GL_NO_ERROR);

	glEnable(GL_DEPTH_TEST);

	//chose a shader
	shader = GFX::Shader::Get("texture");

    assert(glGetError() == GL_NO_ERROR);

	//no shader? then nothing to render
	if (!shader)
		return;
	shader->enable();

	//upload uniforms
	shader->setUniform("u_model", model);
	cameraToShader(camera, shader);
	float t = getTime();
	shader->setUniform("u_time", t );

	shader->setUniform("u_color", material->color);
	if(texture)
		shader->setUniform("u_texture", texture, 0);

	//this is used to say which is the alpha threshold to what we should not paint a pixel on the screen (to cut polygons according to texture alpha)
	shader->setUniform("u_alpha_cutoff", material->alpha_mode == SCN::eAlphaMode::MASK ? material->alpha_cutoff : 0.001f);

	if (render_wireframe)
		glPolygonMode( GL_FRONT_AND_BACK, GL_LINE );

	//do the draw call that renders the mesh into the screen
	mesh->render(GL_TRIANGLES);

	//disable shader
	shader->disable();

	//set the render state as it was before to avoid problems with future renders
	glDisable(GL_BLEND);
	glPolygonMode( GL_FRONT_AND_BACK, GL_FILL );
}

void Renderer::renderMeshWithMaterialPlain(const Matrix44 model, GFX::Mesh* mesh, SCN::Material* material)
{
	//in case there is nothing to do
	if (!mesh || !mesh->getNumVertices() || !material)
		return;
	assert(glGetError() == GL_NO_ERROR);

	//define locals to simplify coding
	GFX::Shader* shader = NULL;
	GFX::Texture* texture = NULL;
	Camera* camera = Camera::current;

	texture = material->textures[SCN::eTextureChannel::ALBEDO].texture;

	if (texture == NULL)
		texture = GFX::Texture::getWhiteTexture(); //a 1x1 white texture

	glDisable(GL_BLEND);

	//select if render both sides of the triangles
	if (material->two_sided)
		glDisable(GL_CULL_FACE);
	else
		glEnable(GL_CULL_FACE);
	assert(glGetError() == GL_NO_ERROR);

	glEnable(GL_DEPTH_TEST);

	//chose a shader
	shader = GFX::Shader::Get("texture");

	assert(glGetError() == GL_NO_ERROR);

	//no shader? then nothing to render
	if (!shader)
		return;
	shader->enable();

	//upload uniforms
	shader->setUniform("u_model", model);
	cameraToShader(camera, shader);

	shader->setUniform("u_color", material->color);
	if (texture)
		shader->setUniform("u_texture", texture, 0);

	//this is used to say which is the alpha threshold to what we should not paint a pixel on the screen (to cut polygons according to texture alpha)
	shader->setUniform("u_alpha_cutoff", material->alpha_mode == SCN::eAlphaMode::MASK ? material->alpha_cutoff : 0.001f);

	//do the draw call that renders the mesh into the screen
	mesh->render(GL_TRIANGLES);

	//disable shader
	shader->disable();
}

void Renderer::renderMeshWithMaterialLights(const Matrix44 model, GFX::Mesh* mesh, SCN::Material* material)
{
	int texPosition = 0;
	int shadowMapPos = 8;
	//in case there is nothing to do
	if (!mesh || !mesh->getNumVertices() || !material )
		return;
    assert(glGetError() == GL_NO_ERROR);

	//define locals to simplify coding
	GFX::Shader* shader = NULL;
	GFX::Texture* texture = NULL;
	Camera* camera = Camera::current;
	
	texture = material->textures[SCN::eTextureChannel::ALBEDO].texture;

	if (use_no_texture)
		texture = GFX::Texture::getWhiteTexture(); //a 1x1 white texture

	if (texture == NULL)
		texture = GFX::Texture::getWhiteTexture(); //a 1x1 white texture

	//select the blending
	if (material->alpha_mode == SCN::eAlphaMode::BLEND)
	{
		glEnable(GL_BLEND);
		glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	}
	else
		glDisable(GL_BLEND);

	//select if render both sides of the triangles
	if(material->two_sided)
		glDisable(GL_CULL_FACE);
	else
		glEnable(GL_CULL_FACE);
    assert(glGetError() == GL_NO_ERROR);

	glEnable(GL_DEPTH_TEST);

	//chose a shader
	shader = GFX::Shader::Get("light");

    assert(glGetError() == GL_NO_ERROR);

	//no shader? then nothing to render
	if (!shader)
		return;
	shader->enable();

	//upload uniforms
	shader->setUniform("u_model", model);
	cameraToShader(camera, shader);
	float t = getTime();
	shader->setUniform("u_time", t );

	//ambient light
	shader->setUniform("u_ambient_light", scene->ambient_light);

	//uniforms to calculate specular: camera position, alpha shininess, specular factor
	shader->setUniform("eye", camera->eye);
	shader->setUniform("alpha", material->roughness_factor);
	float specular_factor = material->metallic_factor;
	shader->setUniform("u_specular", material->metallic_factor);
	shader->setUniform("specular_option", use_specular && specular_factor>0.0f);

	//color
	shader->setUniform("u_color", material->color);
	if(texture)
		shader->setUniform("u_texture", texture, texPosition++);

	//upload of normal, emissive and occlusion (red value of metallic roughness as specified) textures, if they don't exist upload white texture
	GFX::Texture* normalMap    = material->textures[eTextureChannel::NORMALMAP         ].texture;
	GFX::Texture* emissiveTex  = material->textures[eTextureChannel::EMISSIVE          ].texture;
	GFX::Texture* occlusionTex = material->textures[eTextureChannel::METALLIC_ROUGHNESS].texture;

	normalMap    = (normalMap)    ? normalMap    : GFX::Texture::getWhiteTexture();
	emissiveTex  = (emissiveTex)  ? emissiveTex  : GFX::Texture::getWhiteTexture();
	occlusionTex = (occlusionTex) ? occlusionTex : GFX::Texture::getWhiteTexture();

	shader->setUniform("u_normal_texture", normalMap, texPosition++);
	shader->setUniform("normal_option", (int) use_normal_map);

	shader->setUniform("u_emissive_factor", material->emissive_factor);
	shader->setUniform("u_emissive_texture", emissiveTex, texPosition++);
	shader->setUniform("emissive_option", (int) use_emissive);

	shader->setUniform("u_occlusion_texture", occlusionTex, texPosition++);
	shader->setUniform("occlusion_option", (int) use_occlusion);
		
	//this is used to say which is the alpha threshold to what we should not paint a pixel on the screen (to cut polygons according to texture alpha)
	shader->setUniform("u_alpha_cutoff", material->alpha_mode == SCN::eAlphaMode::MASK ? material->alpha_cutoff : 0.001f);

	if (render_wireframe)
		glPolygonMode( GL_FRONT_AND_BACK, GL_LINE );

	if (lights.size()){
		//do the draw call that renders the mesh into the screen
		shader->setUniform("single_pass_option", !use_multipass_lights);
		if (use_multipass_lights)
		{
			if (material->alpha_mode != SCN::eAlphaMode::BLEND)
			{
				glBlendFunc(GL_SRC_ALPHA, GL_ONE);
				glDisable(GL_BLEND);
			}
			glDepthFunc(GL_LEQUAL);

			for (LightEntity* light : lights)
			{
				//upload uniforms and render
				lightToShader(light, shader);

				if (light->cast_shadows && light->shadowMapFBO)
				{
					shadowToShader(light, shadowMapPos, shader);
				}
				else
					shader->setUniform("u_light_cast_shadows", 0);

				if (material->alpha_mode != SCN::eAlphaMode::BLEND || light == lights[0]) mesh->render(GL_TRIANGLES);

				//only upload once
				shader->setUniform("u_ambient_light" , vec3(0.0f));
				shader->setUniform("u_emissive_light", vec3(0.0f));
				glEnable(GL_BLEND);
			}

			glDepthFunc(GL_LESS);
		}
		else
		{
			int n_lights = lights.size();
			
			//upload uniforms and render
			shadowToShader(shader);
			lightToShader(shader);
			mesh->render(GL_TRIANGLES);
		}
	}
	else
	{
		shader->setUniform("u_light_type", 0);
		mesh->render(GL_TRIANGLES);
	}

	//disable shader
	shader->disable();

	//set the render state as it was before to avoid problems with future renders
	glDisable(GL_BLEND);
	glPolygonMode( GL_FRONT_AND_BACK, GL_FILL );
}

void SCN::Renderer::shadowToShader(LightEntity* light, int& shadowMapPos, GFX::Shader* shader)
{
	shader->setUniform("u_light_cast_shadows", 1);
	shader->setUniform("u_shadowmap", light->shadowMapFBO->depth_texture, shadowMapPos++);
	shader->setUniform("u_shadowmap_viewprojection", light->shadowMap_viewProjection);
	shader->setUniform("u_shadow_bias", light->shadow_bias);
}

void SCN::Renderer::shadowToShader(GFX::Shader* shader)
{
	int u_light_cast_shadows_arr[N_LIGHTS];
	mat4 u_shadowmap_viewprojections[N_LIGHTS];
	float u_shadowmap_biases[N_LIGHTS];

	for (int i = 0; i < N_LIGHTS; ++i)
	{
		LightEntity* light = lights[i];
		u_light_cast_shadows_arr[i] = light->cast_shadows;
		u_shadowmap_biases[i] = light->shadow_bias;
	}

	shadowToShaderAuxiliar(shader);

	shader->setUniform1Array("u_light_cast_shadows_arr", (int*) &u_light_cast_shadows_arr, N_LIGHTS);
	shader->setUniform1Array("u_shadowmap_biases", (float*) &u_shadowmap_biases, N_LIGHTS);
}

void SCN::Renderer::shadowToShaderAuxiliar(GFX::Shader* shader)
{
	int shadowMapPos = 8;

	GFX::Texture* shadowTex;

	LightEntity* light = lights[0];
	shadowTex = (light->shadowMapFBO) ? light->shadowMapFBO->depth_texture : GFX::Texture::getWhiteTexture();
	shader->setUniform("u_shadow_textures[0]", shadowTex, shadowMapPos++);
	shader->setUniform("u_light_shadowmap_viewprojections[0]", light->shadowMap_viewProjection);

	light = lights[1];
	shadowTex = (light->shadowMapFBO) ? light->shadowMapFBO->depth_texture : GFX::Texture::getWhiteTexture();
	shader->setUniform("u_shadow_textures[1]", shadowTex, shadowMapPos++);
	shader->setUniform("u_light_shadowmap_viewprojections[1]", light->shadowMap_viewProjection);

	light = lights[2];
	shadowTex = (light->shadowMapFBO) ? light->shadowMapFBO->depth_texture : GFX::Texture::getWhiteTexture();
	shader->setUniform("u_shadow_textures[2]", shadowTex, shadowMapPos++);
	shader->setUniform("u_light_shadowmap_viewprojections[2]", light->shadowMap_viewProjection);

	light = lights[3];
	shadowTex = (light->shadowMapFBO) ? light->shadowMapFBO->depth_texture : GFX::Texture::getWhiteTexture();
	shader->setUniform("u_shadow_textures[3]", shadowTex, shadowMapPos++);
	shader->setUniform("u_light_shadowmap_viewprojections[3]", light->shadowMap_viewProjection);
}

void SCN::Renderer::lightToShader(LightEntity* light, GFX::Shader* shader)
{
	vec2 cone_info = vec2( cosf(light->cone_info.x * PI/180.0f), cosf(light->cone_info.y * PI/180.0f) );

	shader->setUniform("u_light_type"        , (int) light->light_type                    );
	shader->setUniform("u_light_position"    , light->root.model.getTranslation()         );
	shader->setUniform("u_light_front"       , light->root.model.frontVector().normalize());
	shader->setUniform("u_light_color_multi" , light->color*light->intensity              );
	shader->setUniform("u_light_max_distance", light->max_distance                        );
	shader->setUniform("u_light_cone_info"   , cone_info);
}

void SCN::Renderer::lightToShader(GFX::Shader* shader)
{
	vec3 light_position[N_LIGHTS];
	vec3 light_color[N_LIGHTS];
	int light_types[N_LIGHTS];
	float light_max_distances[N_LIGHTS];
	vec2 cone_infos[N_LIGHTS];
	vec3 light_fronts[N_LIGHTS];

	for (int i = 0; i < N_LIGHTS; ++i)
	{
		vec2 cone_info = vec2( cosf(lights[i]->cone_info.x * PI/180.0f), cosf(lights[i]->cone_info.y * PI/180.0f) );

		light_position[i]      = lights[i]->root.model.getTranslation();
		light_color[i]         = lights[i]->color*lights[i]->intensity;
		light_types[i]         = (int) lights[i]->light_type;
		light_max_distances[i] = lights[i]->max_distance;
		cone_infos[i]          = cone_info;
		light_fronts[i]        = lights[i]->root.model.frontVector().normalize();
	}

	shader->setUniform3Array("u_light_pos"          , (float*) &light_position, N_LIGHTS     );
	shader->setUniform3Array("u_light_color"        , (float*) &light_color, N_LIGHTS		 );
	shader->setUniform1Array("u_light_types"        , (int*  ) &light_types, N_LIGHTS		 );
	shader->setUniform1Array("u_light_max_distances", (float*) &light_max_distances, N_LIGHTS);
	shader->setUniform2Array("u_light_cones_info"   , (float*) &cone_infos, N_LIGHTS		 );
	shader->setUniform3Array("u_light_fronts"       , (float*) &light_fronts, N_LIGHTS		 );
	shader->setUniform1("u_num_lights", N_LIGHTS);
}

void SCN::Renderer::cameraToShader(Camera* camera, GFX::Shader* shader)
{
	shader->setUniform("u_viewprojection", camera->viewprojection_matrix );
	shader->setUniform("u_camera_position", camera->eye);
}

#ifndef SKIP_IMGUI

void Renderer::showUI()
{
		
	ImGui::Checkbox("Wireframe", &render_wireframe);
	ImGui::Checkbox("Boundaries", &render_boundaries);

	//add here your stuff
	ImGui::Checkbox("Emissive materials", &use_emissive);
	ImGui::Checkbox("Multipass Light", &use_multipass_lights);
	ImGui::Checkbox("No texture", &use_no_texture);
	ImGui::Checkbox("Normal map", &use_normal_map);
	ImGui::Checkbox("Specular light", &use_specular);
	ImGui::Checkbox("Occlusion light", &use_occlusion);
	ImGui::Checkbox("Create Shadow Maps", &use_shadowMap);

	if (ImGui::Button("ShadowMap 256"))
		shadowmap_size = 256;
	if (ImGui::Button("ShadowMap 512"))
		shadowmap_size = 512;
	if (ImGui::Button("ShadowMap 1024"))
		shadowmap_size = 1024;
	if (ImGui::Button("ShadowMap 2048"))
		shadowmap_size = 2048;

		
}

#else
void Renderer::showUI() {}
#endif