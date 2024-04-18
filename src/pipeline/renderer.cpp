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
	use_emissive = false;
	use_specular = false;
	use_occlusion = false;

	sphere.createSphere(1.0f);
	sphere.uploadToVRAM();
}

void Renderer::extractSceneInfo(SCN::Scene* scene, Camera* camera)
{
	renderables.clear();
	lights.clear();

	for (int i = 0; i < scene->entities.size(); ++i)
	{
		BaseEntity* ent = scene->entities[i];
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
			if ( light->light_type == eLightType::DIRECTIONAL || camera->testSphereInFrustum(light->root.model.getTranslation(), light->max_distance))
				lights.push_back(light);
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

void Renderer::renderScene(SCN::Scene* scene, Camera* camera)
{
	this->scene = scene;
	setupScene();
	extractSceneInfo(scene, camera);

	std::vector<Renderable> opaque;
	std::vector<Renderable> nonOpaque;

	for (Renderable r : renderables)
	{
		if (r.material->alpha_mode == eAlphaMode::NO_ALPHA) opaque.push_back(r);
		else nonOpaque.push_back(r);
	}

	sort(nonOpaque.begin(), nonOpaque.end(), renderableComparator);

	std::vector<Renderable> sortedRenderables;

	sortedRenderables.reserve( opaque.size() + nonOpaque.size() ); // preallocate memory
	sortedRenderables.insert( sortedRenderables.end(), opaque.begin(), opaque.end() );
	sortedRenderables.insert( sortedRenderables.end(), nonOpaque.begin(), nonOpaque.end() );

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
		renderMeshWithMaterialLights(re.model, re.mesh, re.material);
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

void Renderer::renderMeshWithMaterialLights(const Matrix44 model, GFX::Mesh* mesh, SCN::Material* material)
{
	int texPosition = 0;
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

	shader->setUniform("u_ambient_light", scene->ambient_light);

	shader->setUniform("eye", camera->eye);
	shader->setUniform("alpha", material->roughness_factor);

	float specular_factor = NULL; specular_factor = material->metallic_factor;

	if (specular_factor)
	{
		shader->setUniform("u_specular", specular_factor);
		shader->setUniform("specular_option", true);
	}
	else
		shader->setUniform1("specular_option", false);

	shader->setUniform("u_color", material->color);
	if(texture)
		shader->setUniform("u_texture", texture, texPosition++);

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
	if (material->alpha_mode != SCN::eAlphaMode::BLEND)
	{
		glBlendFunc(GL_SRC_ALPHA, GL_ONE);
		glDisable(GL_BLEND);
	}
	glDepthFunc(GL_LEQUAL);


	if (lights.size()){
		//do the draw call that renders the mesh into the screen
		shader->setUniform("single_pass_option", !use_multipass_lights);
		if (use_multipass_lights)
		{
			for (LightEntity* light : lights)
			{

				lightToShader(light, shader);
				mesh->render(GL_TRIANGLES);

				shader->setUniform("u_ambient_light" , vec3(0.0));
				shader->setUniform("u_emissive_light", vec3(0.0f));
				glEnable(GL_BLEND);
			}
		}
		else
		{
			int n_lights = lights.size();

			lightToShader(n_lights, shader);
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

void SCN::Renderer::lightToShader(LightEntity* light, GFX::Shader* shader)
{
	shader->setUniform("u_light_type", (int) light->light_type);
	shader->setUniform("u_light_position", light->root.model.getTranslation());
	shader->setUniform("u_light_color_multi", light->color*light->intensity);
	shader->setUniform("u_light_max_distance", light->max_distance);
	shader->setUniform("u_light_cone_info", vec2( cosf(light->cone_info.x * PI/180.0f), cosf(light->cone_info.y * PI/180.0f) ));
	shader->setUniform("u_light_front", light->root.model.frontVector().normalize());
}

void SCN::Renderer::lightToShader(int n_lights, GFX::Shader* shader)
{
	vec3 light_position[n_lights];
	vec3 light_color[n_lights];
	int light_types[n_lights];
	float light_max_distances[n_lights];
	vec2 cone_infos[n_lights];
	vec3 light_fronts[n_lights];

	for (int i = 0; i < n_lights; ++i)
	{
		light_position[i] = lights[i]->root.model.getTranslation();
		light_color[i] = lights[i]->color*lights[i]->intensity;
		light_types[i] = (int) lights[i]->light_type;
		light_max_distances[i] = lights[i]->max_distance;
		cone_infos[i] = vec2( cosf(lights[i]->cone_info.x * PI/180.0f), cosf(lights[i]->cone_info.y * PI/180.0f) );
		light_fronts[i] = lights[i]->root.model.frontVector().normalize();
	}

	shader->setUniform3Array("u_light_pos", (float*)&light_position, n_lights);
	shader->setUniform3Array("u_light_color", (float*)&light_color, n_lights);
	shader->setUniform1Array("u_light_types", (int*) &light_types, n_lights);
	shader->setUniform1Array("u_light_max_distances", (float*)&light_max_distances, n_lights);
	shader->setUniform2Array("u_light_cones_info", (float*)&cone_infos, n_lights);
	shader->setUniform3Array("u_light_fronts", (float*)&light_fronts, n_lights);
	shader->setUniform1("u_num_lights", n_lights);
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
}

#else
void Renderer::showUI() {}
#endif