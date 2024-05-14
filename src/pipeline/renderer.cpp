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
GFX::FBO* gbuffers = nullptr;
GFX::FBO* illumination_fbo = nullptr;
GFX::FBO*  ssao_fbo = nullptr;
GFX::FBO* final_fbo = nullptr;

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
	use_normal_map = true;
	use_emissive   = true;
	use_specular   = true;
	use_occlusion  = true;
	view_ssao = false;
	use_ssao = false;
	use_degamma = false;

	sphere.createSphere(1.0f);
	sphere.uploadToVRAM();

	shadowmap_size = 1024;

	shadowmap_size = 1024;
	mainLight = nullptr;

	pipeline_mode = ePipelineMode::DEFERRED;
	gbuffer_show_mode = eShowGBuffer::NONE;

	ssao_radius = 11.0f;
	ssao_max_distance = 0.1f;
	random_points = generateSpherePoints(64, 1, false);
}
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
			//pos = camera->eye;
			light_camera.lookAt(pos, pos - light->getFront(), up);

			float halfArea = light->area / 2.0f;
			light_camera.setOrthographic(-halfArea, halfArea, -halfArea, halfArea, light->near_distance, light->max_distance);

			//compute texel size in world units, where frustum size is the distance from left to right in the camera
			float grid = light->area / (float) shadowmap_size;

			//snap camera X,Y to that size in camera space assuming the frustum is square, otherwise compute gridx and gridy
				light_camera.view_matrix.M[3][0] = round(light_camera.view_matrix.M[3][0] / grid) * grid;

			light_camera.view_matrix.M[3][1] = round(light_camera.view_matrix.M[3][1] / grid) * grid;

			//update viewproj matrix (be sure no one changes it)
			light_camera.viewprojection_matrix = light_camera.view_matrix * light_camera.projection_matrix;

		}
		else if (light->light_type == eLightType::SPOT)
		{
			light_camera.setPerspective(light->cone_info.y * 2.0f, 1.0f, light->near_distance, light->max_distance);
		}

		light->shadowMap_viewProjection = light_camera.viewprojection_matrix;
		light_camera.enable();

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

	if (pipeline_mode == ePipelineMode::FORWARD)
		renderSceneForward(scene, camera);
	else if (pipeline_mode == ePipelineMode::DEFERRED)
		renderSceneDeferred(scene, camera);
}

void Renderer::renderSceneForward(SCN::Scene* scene, Camera* camera)
{
	camera->enable();
	glDisable(GL_BLEND);
	glEnable(GL_DEPTH_TEST);

	//set the clear color (the background color)
	glClearColor(scene->background_color.x, scene->background_color.y, scene->background_color.z, 1.0);

	// Clear the color and the depth buffer
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	GFX::checkGLErrors();


	//render skybox
	if (skybox_cubemap)
		renderSkybox(skybox_cubemap);

	sort(alpha_renderables.begin(), alpha_renderables.end(), renderableComparator);

	std::vector<Renderable> sortedRenderables;

	sortedRenderables.reserve(opaque_renderables.size() + alpha_renderables.size()); // preallocate memory
	sortedRenderables.insert(sortedRenderables.end(), opaque_renderables.begin(), opaque_renderables.end());
	sortedRenderables.insert(sortedRenderables.end(), alpha_renderables.begin(), alpha_renderables.end());

	//render entities
	for (Renderable& re : sortedRenderables)
	{
		if (camera->testBoxInFrustum(re.bounding.center, re.bounding.halfsize))
			renderMeshWithMaterialLights(re.model, re.mesh, re.material);
	}

}

void Renderer::gbufferToShader(GFX::Shader* shader, vec2 size, Camera* camera)
{
	int texturePos = 0;
	shader->setUniform("u_color_texture",    gbuffers->color_textures[0], texturePos++);
	shader->setUniform("u_normal_texture",	 gbuffers->color_textures[1], texturePos++);
	shader->setUniform("u_extra_texture",    gbuffers->color_textures[2], texturePos++);
	shader->setUniform("u_metallic_texture", gbuffers->color_textures[3], texturePos++);
	shader->setUniform("u_depth_texture",    gbuffers->depth_texture,	  texturePos++);
	shader->setUniform("u_cube_texture",     skybox_cubemap,			  texturePos++);
	cameraToShader(camera, shader);
	shader->setUniform("u_iRes", vec2(1.0 / size.x, 1.0 / size.y));
	shader->setUniform("u_inverse_viewprojection", camera->inverse_viewprojection_matrix);
	shader->setUniform("specular_option",  (int) use_specular);
	shader->setUniform("u_use_ssao", (int)use_ssao);
	if (use_ssao)
		shader->setUniform("u_ao_texture", ssao_fbo->color_textures[0], texturePos++);
}


void Renderer::renderSceneDeferred(SCN::Scene* scene, Camera* camera)
{
	vec2 size = CORE::getWindowSize();
	int shadowMapPos = 8;
	GFX::Mesh* quad = GFX::Mesh::getQuad();

	//generate gbuffers
	if (!gbuffers)
	{
		gbuffers = new GFX::FBO();
		gbuffers->create(size.x/2, size.y/2, 4, GL_RGBA, GL_UNSIGNED_BYTE, true);  //crea todas las texturas attached, true if we want depthbuffer in a texure (para poder leerlo)
		
	}

	gbuffers->bind();

	camera->enable();
	glEnable(GL_DEPTH_TEST);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	//set the clear color
	glClearColor(0, 0, 0, 1.0f);

	for (Renderable& re : opaque_renderables)
		if (camera->testBoxInFrustum(re.bounding.center, re.bounding.halfsize))
			renderMeshWithMaterialGBuffers(re.model, re.mesh, re.material);

	gbuffers->unbind();



	//ssao
	if (!ssao_fbo) {
		ssao_fbo = new GFX::FBO();
		ssao_fbo->create(size.x, size.y, 1, GL_RGB, GL_UNSIGNED_BYTE, false);
		ssao_fbo->color_textures[0]->setName("SSAO");
	}

	ssao_fbo->bind();
	glClearColor(1, 1, 1, 1); //fondo blanco
	glClear(GL_COLOR_BUFFER_BIT);
	glDisable(GL_DEPTH_TEST);
	glDisable(GL_BLEND);

	GFX::Shader* sh_ssao = GFX::Shader::Get("ssao");
	assert(sh_ssao);
	sh_ssao->enable();
	sh_ssao->setUniform("u_depth_texture", gbuffers->depth_texture, 0);

	sh_ssao->setUniform("u_radius", ssao_radius);
	sh_ssao->setUniform("far", ssao_max_distance);
	sh_ssao->setUniform("near", (float)0.0001f);

	//to reconstruct world position
	sh_ssao->setUniform("u_iRes", vec2(1.0 / ssao_fbo->color_textures[0]->width, 1.0 / ssao_fbo->color_textures[0]->height));
	sh_ssao->setUniform("u_inverse_viewprojection", camera->inverse_viewprojection_matrix);
	sh_ssao->setUniform3Array("u_points", (float*)&random_points[0], random_points.size());
	sh_ssao->setUniform("u_viewprojection", camera->viewprojection_matrix);

	quad->render(GL_TRIANGLES);


	ssao_fbo->unbind();

	//interpolacion
	//bind the texture we want to change 
	ssao_fbo->color_textures[0]->bind();
	//disable using mipmaps
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	//enable bilinear filtering
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

	ssao_fbo->color_textures[0]->unbind();


	//ilumination pass
	if (!illumination_fbo) {
		illumination_fbo = new GFX::FBO();
		illumination_fbo->create(size.x, size.y, 1, GL_RGB, GL_FLOAT, false);
	}
	
	illumination_fbo->bind();

	glClearColor(scene->background_color.x, scene->background_color.y, scene->background_color.z, 1.0);
	glClearColor(0, 0, 0, 1.0f);//set the clear color
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	if (skybox_cubemap)
		renderSkybox(skybox_cubemap);


	GFX::Shader* shader = GFX::Shader::Get("deferred_global");

	glDisable(GL_BLEND);
	glDisable(GL_DEPTH_TEST);
	shader->enable();

	if (mainLight->cast_shadows && mainLight->shadowMapFBO)
	{
		shadowToShader(mainLight, shadowMapPos, shader);
	}
	else
		shader->setUniform("u_light_cast_shadows", 0);

	gbufferToShader(shader, size, camera);

	shader->setUniform("u_ambient_light", scene->ambient_light);
	shader->setUniform("use_degamma", (int)use_degamma);

	lightToShader(mainLight, shader);

	quad->render(GL_TRIANGLES);

	shader = GFX::Shader::Get("deferred_ws");
	shader->enable();

	gbufferToShader(shader, size, camera);
	shader->setUniform("use_degamma", (int)use_degamma);


	glDepthFunc(GL_GREATER);
	glEnable(GL_BLEND);
	glEnable(GL_CULL_FACE);
	glFrontFace(GL_CW);
	glBlendFunc(GL_ONE, GL_ONE);
	glDepthMask(false);

	for (auto light : lights) {
		if (light->light_type == eLightType::POINT || light->light_type == eLightType::SPOT) {
			if (light->cast_shadows)
			{
				shadowToShader(light, shadowMapPos, shader);
			}
			else shader->setUniform("u_light_cast_shadows", 0);

			Matrix44 model;
			vec3 lightpos = light->getGlobalPosition();
			model.translate(lightpos.x, lightpos.y, lightpos.z);
			model.scale(light->max_distance, light->max_distance, light->max_distance);
			shader->setUniform("u_model", model);

			lightToShader(light, shader);
			sphere.render(GL_TRIANGLES);
		}
	}

	shader->disable();
	glDisable(GL_BLEND);
	glFrontFace(GL_CCW);
	glDisable(GL_CULL_FACE);
	glDepthFunc(GL_LESS);
	glDepthMask(true);
	illumination_fbo->unbind();

	illumination_fbo->color_textures[0]->toViewport();

	//degamma final pass
	/*if (!final_fbo) {
		final_fbo = new GFX::FBO();
		final_fbo->create(size.x, size.y, 1, GL_RGB, GL_FLOAT, false);
	}

	GFX::Shader* sh_gamma = GFX::Shader::Get("gamma");
	assert(sh_gamma);

	final_fbo->bind();
	sh_gamma->setUniform("u_texture", illumination_fbo->color_textures[0], 0);
	//quad->render(GL_TRIANGLES);
	final_fbo->unbind();

	final_fbo->color_textures[0]->toViewport();
	*/


	//alpha renderables
	//sort(alpha_renderables.begin(), alpha_renderables.end(), renderableComparator);
	//for (Renderable& re : alpha_renderables)
	//{
	//	if (camera->testBoxInFrustum(re.bounding.center, re.bounding.halfsize))
	//		renderMeshWithMaterialLights(re.model, re.mesh, re.material);
	//}


	glDisable(GL_DEPTH_TEST);
	switch (gbuffer_show_mode) //debug
	{
		case eShowGBuffer::COLOR:  gbuffers->color_textures[0]->toViewport(); break;
		case eShowGBuffer::NORMAL: gbuffers->color_textures[1]->toViewport(); break;
		case eShowGBuffer::EXTRA:  gbuffers->color_textures[2]->toViewport(); break;
		case eShowGBuffer::DEPTH:  gbuffers->depth_texture->toViewport();	  break; //para visualizar depth usar depth.fs y funcion
	}

	if (view_ssao)
		ssao_fbo->color_textures[0]->toViewport();
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

void Renderer::renderMeshWithMaterialGBuffers(const Matrix44 model, GFX::Mesh* mesh, SCN::Material* material)
{
	int texturePosition = 0;
	//in case there is nothing to do
	if (!mesh || !mesh->getNumVertices() || !material)
		return;
	assert(glGetError() == GL_NO_ERROR);

	//define locals to simplify coding
	GFX::Shader* shader = NULL;
	Camera* camera = Camera::current;

	GFX::Texture* texture = material->textures[SCN::eTextureChannel::ALBEDO].texture;
	GFX::Texture* metallic_roughness_texture = material->textures[eTextureChannel::METALLIC_ROUGHNESS].texture;
	GFX::Texture* emissive_texture = material->textures[eTextureChannel::EMISSIVE].texture;
	GFX::Texture* normal_texture = material->textures[eTextureChannel::NORMALMAP].texture;

	emissive_texture = (emissive_texture) ? emissive_texture : GFX::Texture::getWhiteTexture();
	metallic_roughness_texture = (metallic_roughness_texture) ? metallic_roughness_texture : GFX::Texture::getWhiteTexture();

	if (texture == NULL || use_no_texture)
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
	shader = GFX::Shader::Get("gbuffers");

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
		shader->setUniform("u_texture", texture, texturePosition++);

	//added
	if (metallic_roughness_texture)
		shader->setUniform("u_metallic_roughness_texture", metallic_roughness_texture, texturePosition++);

	if (emissive_texture)
		shader->setUniform("u_emissive_texture", emissive_texture, texturePosition++);

	shader->setUniform("u_emissive_factor", material->emissive_factor);

	float specular_factor = material->metallic_factor;
	shader->setUniform("u_metallic_factor", material->metallic_factor);
	shader->setUniform("u_metallic_roughness", material->roughness_factor);

	if (!normal_texture)
		shader->setUniform("normal_option", 0);
	else
	{
		shader->setUniform("u_normal_texture", normal_texture, texturePosition++);
		shader->setUniform("normal_option", (int) use_normal_map);
	}

	shader->setUniform("occlusion_option", (int)use_occlusion);
	shader->setUniform("emissive_option",  (int)use_emissive);

	//this is used to say which is the alpha threshold to what we should not paint a pixel on the screen (to cut polygons according to texture alpha)
	shader->setUniform("u_alpha_cutoff", material->alpha_mode == SCN::eAlphaMode::MASK ? material->alpha_cutoff : 0.001f);

	//do the draw call that renders the mesh into the screen
	mesh->render(GL_TRIANGLES);

	//disable shader
	shader->disable();
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
	if (texture == NULL || use_no_texture)
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
	shader = (use_multipass_lights) ? GFX::Shader::Get("multipass_light") : GFX::Shader::Get("singlepass_light");
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
	
	emissiveTex  = (emissiveTex)  ? emissiveTex  : GFX::Texture::getWhiteTexture();
	occlusionTex = (occlusionTex) ? occlusionTex : GFX::Texture::getWhiteTexture();

	if (!normalMap)
		shader->setUniform("normal_option", (int) 0);
	else {
		shader->setUniform("u_normal_texture", normalMap, texPosition++);
		shader->setUniform("normal_option", (int) use_normal_map);
	}

	shader->setUniform("u_emissive_factor", material->emissive_factor);
	shader->setUniform("u_emissive_texture", emissiveTex, texPosition++);
	shader->setUniform("emissive_option", (int) use_emissive);

	shader->setUniform("u_metallic_roughness_texture", occlusionTex, texPosition++);
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
			shader = GFX::Shader::Get("multipass_light");
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
				shader->setUniform("u_emissive_factor", vec3(0.0f));
				glEnable(GL_BLEND);
			}

			glDepthFunc(GL_LESS);
		}
		else
		{
			shader = GFX::Shader::Get("singlepass_light");
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

	shader->setUniform1Array("u_light_cast_shadows", (int*) &u_light_cast_shadows_arr, N_LIGHTS);
	shader->setUniform1Array("u_shadowmap_bias"    , (float*) &u_shadowmap_biases, N_LIGHTS);
}

void SCN::Renderer::shadowToShaderAuxiliar(GFX::Shader* shader)
{
	int shadowMapPos = 8;

	GFX::Texture* shadowTex;

	LightEntity* light = lights[0];
	shadowTex = (light->shadowMapFBO) ? light->shadowMapFBO->depth_texture : GFX::Texture::getWhiteTexture();
	shader->setUniform("u_shadow_texture[0]", shadowTex, shadowMapPos++);
	shader->setUniform("u_light_shadowmap_viewprojection[0]", light->shadowMap_viewProjection);

	light = lights[1];
	shadowTex = (light->shadowMapFBO) ? light->shadowMapFBO->depth_texture : GFX::Texture::getWhiteTexture();
	shader->setUniform("u_shadow_texture[1]", shadowTex, shadowMapPos++);
	shader->setUniform("u_light_shadowmap_viewprojection[1]", light->shadowMap_viewProjection);

	light = lights[2];
	shadowTex = (light->shadowMapFBO) ? light->shadowMapFBO->depth_texture : GFX::Texture::getWhiteTexture();
	shader->setUniform("u_shadow_texture[2]", shadowTex, shadowMapPos++);
	shader->setUniform("u_light_shadowmap_viewprojection[2]", light->shadowMap_viewProjection);

	light = lights[3];
	shadowTex = (light->shadowMapFBO) ? light->shadowMapFBO->depth_texture : GFX::Texture::getWhiteTexture();
	shader->setUniform("u_shadow_texture[3]", shadowTex, shadowMapPos++);
	shader->setUniform("u_light_shadowmap_viewprojection[3]", light->shadowMap_viewProjection);
}

void SCN::Renderer::lightToShader(LightEntity* light, GFX::Shader* shader)
{
	vec2 cone_info = vec2( cosf(light->cone_info.x * PI/180.0f), cosf(light->cone_info.y * PI/180.0f) );

	shader->setUniform("u_light_type"        , (int) light->light_type                    );
	shader->setUniform("u_light_position"    , light->root.model.getTranslation()         );
	shader->setUniform("u_light_front"       , light->root.model.frontVector().normalize());
	shader->setUniform("u_light_color"       , light->color*light->intensity              );
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
		LightEntity* light = lights[i];

		vec2 cone_info = vec2( cosf(light->cone_info.x * PI/180.0f), cosf(light->cone_info.y * PI/180.0f) );

		light_position[i]      = light->getGlobalPosition();
		light_color[i]         = light->color*lights[i]->intensity;
		light_types[i]         = (int) light->light_type;
		light_max_distances[i] = light->max_distance;
		cone_infos[i]          = cone_info;
		light_fronts[i]        = light->root.model.frontVector().normalize();
	}

	shader->setUniform3Array("u_light_pos"          , (float*) &light_position, N_LIGHTS     );
	shader->setUniform3Array("u_light_color"        , (float*) &light_color, N_LIGHTS		 );
	shader->setUniform1Array("u_light_type"        , (int*  ) &light_types, N_LIGHTS		 );
	shader->setUniform1Array("u_light_max_distance" , (float*) &light_max_distances, N_LIGHTS);
	shader->setUniform2Array("u_light_cone_info"    , (float*) &cone_infos, N_LIGHTS		 );
	shader->setUniform3Array("u_light_front"        , (float*) &light_fronts, N_LIGHTS		 );
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

	ImGui::Combo("Pipeline", (int*)&pipeline_mode, "FLAT\0FORWARD\0DEFERRED\0", ePipelineMode::PIPELINECOUNT);
	ImGui::Combo("Show GBuffer", (int*)&gbuffer_show_mode, "NONE\0COLOR\0NORMAL\0EXTRA\0DEPTH\0", eShowGBuffer::SHOW_BUFFER_COUNT);

	//add here your stuff
	ImGui::Checkbox("Emissive materials", &use_emissive);
	ImGui::Checkbox("Multipass Light", &use_multipass_lights);
	ImGui::Checkbox("No texture", &use_no_texture);
	ImGui::Checkbox("Normal map", &use_normal_map);
	ImGui::Checkbox("Specular light", &use_specular);
	ImGui::Checkbox("Occlusion light", &use_occlusion);

	ImGui::Checkbox("Use SSAO+", &use_ssao);
	ImGui::DragFloat("ssao radius", &ssao_radius,0.01f,0.0f);
	ImGui::DragFloat("ssao max distance", &ssao_max_distance, 0.01f, 0.0f);
	ImGui::Checkbox("View ssao", &view_ssao);

	ImGui::Checkbox("View degamma", &use_degamma);


	
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