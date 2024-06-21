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
Renderer* Renderer::instance = NULL;

GFX::Mesh sphere;
GFX::FBO* gbuffers = nullptr;
GFX::FBO* illumination_fbo = nullptr;
GFX::FBO* ssao_fbo = nullptr;
GFX::FBO* final_fbo = nullptr;
GFX::FBO* blur_fbo = nullptr;

GFX::FBO* irr_fbo = nullptr;
GFX::FBO* renderFBO = nullptr;

GFX::FBO* probe_illumination_fbo = nullptr;
GFX::FBO* combined_illumination_fbo = nullptr;

GFX::Texture* cloned_depth_texture = nullptr;
GFX::FBO* reflections_fbo = nullptr;

GFX::FBO* postfx_fbo = nullptr;
GFX::FBO* postfxIN_fbo = nullptr;
GFX::FBO* postfxOUT_fbo = nullptr;

sProbe probe;
std::vector<sProbe> probes;

sIrradianceInfo probes_info;    //a place to store info about the layout of the grid

GFX::Texture* probes_texture = nullptr;

std::vector< sReflectionProbe> reflection_probes; //tres y buscar cu�l es la que esta m�s cerca !! to implement

sReflectionProbe reflection_probe;
GFX::Mesh cube;

//post
GFX::FBO* dof_fbo = nullptr;
GFX::FBO* blur_final_fbo = nullptr;
sReflectionProbe refl_probe;


float gamma = 2.2;

//volumetrics
GFX::FBO* volumetric_fbo = nullptr;

Renderer::Renderer(const char* shader_atlas_filename)
{
	Renderer::instance = this;

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
	use_emissive = true;
	use_specular = true;
	use_occlusion = true;
	view_ssao = false;
	use_ssao = false;
	use_degamma = false;
	view_blur = false;
	use_blur = false;
	use_dithering = false;
	use_volumetric = false;
	constant_density = false;
	use_vignetting = false;
	use_color_correction = false;
	use_chromatic_aberration = false;
	use_fish_eye = true;

	use_lut = false;
	use_lut2 = false;

	lut_amount = 0.2f;
	lut_amount2 = 0.2f;

	use_dof = false;
	show_probes = false;
	combined_irr = false;
	render_refelction_probes = false;

	use_simpleblurr = true;

	ca_strength = 15.0f;
	vignetting = 1.2f;
	contrast = vec3(1.0f);
	fish_eye_strength = 0.5f;

	sphere.createSphere(1.0f);
	sphere.uploadToVRAM();


	cube.createCube(vec3(5.0, 5.0, 5.0));
	cube.uploadToVRAM();


	shadowmap_size = 1024;
	mainLight = nullptr;

	pipeline_mode = ePipelineMode::DEFERRED;
	gbuffer_show_mode = eShowGBuffer::NONE;

	ssao_radius = 11.0f;
	ssao_max_distance = 0.1f;
	random_points = generateSpherePoints(64, 1, false);

	df_min_distance = 1.0f;
	df_max_distance = 3.0f;
	df_scale_blur = 1.0f;
	df_focal_distance = 1.5f;

	//tonemapper attributes
	use_tonemapper = false;
	curr_tonemapper = 0.0f;
	tonemapper_scale = 1.0f;
	tonemapper_avg_lum = 1.0f;
	tonemapper_lumwhite = 1.0f;
	air_density = 0.0001;

	//define grid proves-------------------------------
	
	

	//define grid proves
	//define bounding of the grid and num probes
	probes_info.start.set(-400, 0, -400); //sc1
	probes_info.end.set(400, 300, 400); //sc1

	

	//probes_info.start.set(80, 0, 90); //sc2
	//probes_info.end.set(-80, 80, -90); //sc2


	probes_info.dim.set(11, 5, 11);

	//compute the vector from one corner to the other
	vec3 delta = (probes_info.end - probes_info.start);
	//compute delta from one probe to the next one
	delta.x /= (probes_info.dim.x - 1);
	delta.y /= (probes_info.dim.y - 1);
	delta.z /= (probes_info.dim.z - 1);
	probes_info.delta = delta;

	//store grid


	for (int z = 0; z < probes_info.dim.z; ++z) {
		for (int y = 0; y < probes_info.dim.y; ++y) {
			for (int x = 0; x < probes_info.dim.x; ++x)
			{
				sProbe p;
				p.local.set(x, y, z);

				//index in the linear array
				p.index = x + y * probes_info.dim.x + z *
					probes_info.dim.x * probes_info.dim.y;

				//and its position
				p.pos = probes_info.start +
					probes_info.delta * Vector3f(x, y, z);
				probes.push_back(p);
			}
		}
	}


	// saveIrradianceToDisk ---------------------------------

	//write to file header and probes data
	FILE* f = fopen("irradiance.bin", "wb");
	fwrite(&probes_info, sizeof(sIrradianceInfo), 1, f);
	fwrite(&(probes[0]), sizeof(sProbe), probes.size(), f);
	fclose(f);

	//reflections--------------------------------------------

	//1-probe set it up
	refl_probe.pos.set(-80, 16, 43);
	refl_probe.cubemap = new GFX::Texture();
	refl_probe.cubemap->createCubemap(
		512, 512, 	//size
		NULL, 	//data
		GL_RGB, GL_UNSIGNED_INT, true);	//mipmaps

	//add it to the list
	reflection_probes.push_back(refl_probe);


	//2- probe set it up
	refl_probe.pos.set(0, 100, 300);
	refl_probe.cubemap = new GFX::Texture();
	refl_probe.cubemap->createCubemap(
		512, 512, 	//size
		NULL, 	//data
		GL_RGB, GL_UNSIGNED_INT, true);	//mipmaps

	//add it to the list
	reflection_probes.push_back(refl_probe);



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
			re.mesh = node->mesh;
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
	//decals.clear();
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
			PrefabEntity* pent = (SCN::PrefabEntity*)ent;
			if (pent->prefab)
				extractRenderables(&pent->root, camera);
		}
		else if (ent->getType() == eEntityType::LIGHT)
		{
			LightEntity* light = (SCN::LightEntity*)ent;

			mat4 globalMatrix = light->root.getGlobalMatrix();

			if (light->light_type == eLightType::DIRECTIONAL || camera->testSphereInFrustum(globalMatrix.getTranslation(), light->max_distance))
				lights.push_back(light);

			if (!mainLight && light->light_type == eLightType::DIRECTIONAL)
				mainLight = light;

		}
		/*else if (ent->getType() == eEntityType::DECAL)
				decals.push_back((DecalEntity*)ent);*/
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
			float grid = light->area / (float)shadowmap_size;

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
	vec2 size = CORE::getWindowSize();

	if (!illumination_fbo)
	{
		illumination_fbo = new GFX::FBO();
		illumination_fbo->create(size.x, size.y, 1, GL_RGB, GL_FLOAT, false);
	}

	//illumination_fbo->bind();

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

	/*illumination_fbo->unbind();

	if (use_degamma)
		illumination_fbo->color_textures[0]->toViewport(GFX::Shader::Get("gamma"));

	else
		illumination_fbo->color_textures[0]->toViewport();*/
}

void Renderer::gbufferToShader(GFX::Shader* shader, vec2 size, Camera* camera)
{
	int texturePos = 0;
	shader->setUniform("u_color_texture", gbuffers->color_textures[0], texturePos++);
	shader->setUniform("u_normal_texture", gbuffers->color_textures[1], texturePos++);
	shader->setUniform("u_extra_texture", gbuffers->color_textures[2], texturePos++);
	shader->setUniform("u_metallic_texture", gbuffers->color_textures[3], texturePos++);
	shader->setUniform("u_depth_texture", gbuffers->depth_texture, texturePos++);
	shader->setUniform("u_cube_texture", skybox_cubemap, texturePos++);
	cameraToShader(camera, shader);
	shader->setUniform("u_iRes", vec2(1.0 / size.x, 1.0 / size.y));
	shader->setMatrix44("u_inverse_viewprojection", camera->inverse_viewprojection_matrix);
	shader->setUniform("specular_option", (int)use_specular);
	shader->setUniform("u_use_ssao", (use_ssao || use_blur));

	if (use_ssao)
		shader->setUniform("u_ao_texture", ssao_fbo->color_textures[0], texturePos++);
	else if (use_blur)
		shader->setUniform("u_ao_texture", blur_fbo->color_textures[0], texturePos++);

	shader->setUniform("dithering_option", (int)use_dithering);
}

void Renderer::lightsDeferred(Camera* camera)
{
	int shadowMapPos = 8;
	vec2 size = CORE::getWindowSize();

	GFX::Mesh* quad = GFX::Mesh::getQuad();
	GFX::Shader* shader = GFX::Shader::Get("deferred_global");

	glDisable(GL_BLEND);
	glDisable(GL_DEPTH_TEST);
	shader->enable();

	if (mainLight) {
		if (mainLight->cast_shadows && mainLight->shadowMapFBO)
		{
			shadowToShader(mainLight, shadowMapPos, shader);
		}
		else
			shader->setUniform("u_light_cast_shadows", 0);

		lightToShader(mainLight, shader);
	}
	else shader->setUniform("u_light_type", 0);

	gbufferToShader(shader, size, camera);
	shader->setUniform("u_ambient_light", scene->ambient_light);

	quad->render(GL_TRIANGLES);

	shader = GFX::Shader::Get("deferred_ws");
	shader->enable();

	gbufferToShader(shader, size, camera);

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
			shader->setUniform("u_ambient_light", vec3(0));

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
}

void Renderer::ssaoBlur(Camera* camera)
{
	vec2 size = CORE::getWindowSize();
	GFX::Mesh* quad = GFX::Mesh::getQuad();

	//ssao
	if (!ssao_fbo) {
		ssao_fbo = new GFX::FBO();
		ssao_fbo->create(size.x, size.y, 1, GL_RGB, GL_UNSIGNED_BYTE, true);
		ssao_fbo->color_textures[0]->setName("SSAO");
	}

	ssao_fbo->bind();
	glClearColor(1, 1, 1, 1); //fondo blanco
	glClear(GL_COLOR_BUFFER_BIT);
	glDisable(GL_DEPTH_TEST);
	glDisable(GL_BLEND);

	GFX::Shader* sh_ssao = GFX::Shader::Get("ssao");
	assert(sh_ssao);

	//bind the texture we want to change 
	gbuffers->depth_texture->bind();
	//disable using mipmaps
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	//enable bilinear filtering
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

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

	//blur
	if (!blur_fbo) {
		blur_fbo = new GFX::FBO();
		blur_fbo->create(size.x, size.y, 1, GL_RGB, GL_UNSIGNED_BYTE, false);
		blur_fbo->color_textures[0]->setName("blur");
	}
	blur_fbo->bind();

	glClearColor(1, 1, 1, 1); //fondo blanco
	glClear(GL_COLOR_BUFFER_BIT);
	glDisable(GL_DEPTH_TEST);
	glDisable(GL_BLEND);

	GFX::Shader* sh_blur = GFX::Shader::Get("blur");
	assert(sh_blur);

	sh_blur->enable();

	sh_blur->setUniform("u_ssao_texture", ssao_fbo->color_textures[0], 1);
	sh_blur->setUniform("u_iRes", vec2(1.0 / blur_fbo->color_textures[0]->width, 1.0 / blur_fbo->color_textures[0]->height));

	quad->render(GL_TRIANGLES);

	blur_fbo->unbind();
}

void Renderer::renderTonemapper(GFX::Texture* colorTex)
{
	GFX::Shader* shader;
	//TONEMAPPER

	shader = GFX::Shader::Get("tonemapper");
	shader->enable();
	shader->setUniform("u_scale", tonemapper_scale);
	shader->setUniform("u_average_lum", tonemapper_avg_lum);
	shader->setUniform("u_lumwhite2", tonemapper_lumwhite);
	shader->setUniform("u_igamma", 1.0f / gamma);

	colorTex->toViewport(shader);
	shader->disable();
}

void Renderer::renderIlluminationScene(Camera* camera, vec2* size) {
	if (!illumination_fbo)
	{
		illumination_fbo = new GFX::FBO();
		illumination_fbo->create(size->x, size->y, 1, GL_RGB, GL_FLOAT, false);
	}

	illumination_fbo->bind();

	gbuffers->depth_texture->copyTo(NULL);

	glClearColor(scene->background_color.x, scene->background_color.y, scene->background_color.z, 1.0);
	glClearColor(0, 0, 0, 1.0f);//set the clear color
	glClear(GL_COLOR_BUFFER_BIT);

	if (skybox_cubemap)
		renderSkybox(skybox_cubemap);

	lightsDeferred(camera);


	illumination_fbo->unbind();

}

void Renderer::renderIrradianceScene(Camera* camera,vec2* size)
{


	if (!probe_illumination_fbo)
	{
		probe_illumination_fbo = new GFX::FBO();
		probe_illumination_fbo->create(size->x, size->y, 1, GL_RGB, GL_FLOAT, false);
		probe_illumination_fbo->color_textures[0]->setName("irradiance_probe");

	}


	// Render probe irradiance
	probe_illumination_fbo->bind();
	glClear(GL_COLOR_BUFFER_BIT);


	if (probes_texture) {


		GFX::Mesh* quad = GFX::Mesh::getQuad();

		GFX::Shader* sh_irradiance = GFX::Shader::Get("irradiance");
		assert(sh_irradiance);

		sh_irradiance->enable();


		sh_irradiance->setUniform("u_irr_start", probes_info.start);
		sh_irradiance->setUniform("u_irr_end", probes_info.end);
		sh_irradiance->setUniform("u_irr_normal_distance", (float)0.0f);
		sh_irradiance->setUniform("u_irr_delta", probes_info.delta);
		sh_irradiance->setUniform("u_irr_dims", probes_info.dim);

		probes_info.num_probes = probes.size();

		sh_irradiance->setUniform("u_num_probes", (int)probes_info.num_probes);

		int texturePos = 0;

		sh_irradiance->setUniform("u_probes_texture", probes_texture, texturePos++);

		// you need also pass the distance factor, for now leave it as 0.0
		sh_irradiance->setUniform("u_irr_normal_distance", 0.0f);

		//gbufers
		sh_irradiance->setUniform("u_color_texture", gbuffers->color_textures[0], texturePos++);
		sh_irradiance->setUniform("u_normal_texture", gbuffers->color_textures[1], texturePos++);
		sh_irradiance->setUniform("u_extra_texture", gbuffers->color_textures[2], texturePos++);
		sh_irradiance->setUniform("u_metallic_texture", gbuffers->color_textures[3], texturePos++);
		sh_irradiance->setUniform("u_depth_texture", gbuffers->depth_texture, texturePos++);

		//to reconstruct world position
		sh_irradiance->setUniform("u_iRes", vec2(1.0 / size->x, 1.0 / size->y));
		sh_irradiance->setUniform("u_inverse_viewprojection", camera->inverse_viewprojection_matrix);
		sh_irradiance->setUniform("u_viewprojection", camera->viewprojection_matrix);

		quad->render(GL_TRIANGLES);
		sh_irradiance->disable();


	}


	probe_illumination_fbo->unbind();
}

void Renderer::renderSceneDeferred(SCN::Scene* scene, Camera* camera)
{
	vec2 size = CORE::getWindowSize();
	int shadowMapPos = 8;

	//generate gbuffers
	if (!gbuffers)
	{
		gbuffers = new GFX::FBO();
		gbuffers->create(size.x, size.y, 4, GL_RGBA, GL_UNSIGNED_BYTE, true);  //crea todas las texturas attached, true if we want depthbuffer in a texure (para poder leerlo)
	}

	gbuffers->bind();

	camera->enable();
	glEnable(GL_DEPTH_TEST);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	//set the clear color
	glClearColor(0, 0, 0, 1.0f);

	std::vector<Renderable> toRender = (use_dithering) ? renderables : opaque_renderables;

	for (Renderable& re : toRender)
		if (camera->testBoxInFrustum(re.bounding.center, re.bounding.halfsize))
			renderMeshWithMaterialGBuffers(re.model, re.mesh, re.material);

	gbuffers->unbind();

	//DECALS (prueba para un decal) -------------------------------------------------------------------------------------------------------------------------------------------------------
	
	//if (!cloned_depth_texture)
		//cloned_depth_texture = new GFX::Texture(size.x, size.y, GL_DEPTH_COMPONENT, GL_UNSIGNED_INT, false);


	//copy gbuffers depth to other texture
	//gbuffers->depth_texture->copyTo(cloned_depth_texture);


	//enable alpha blending
	//glDisable(GL_BLEND);
	//glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	//block changing alpha channel
	//glColorMask(true, true, true, false);

	//block from writing to normalmap gbuffer
	//gbuffers->enableBuffers(true, false, false, false); //normalmap our second pos

	/*glEnable(GL_DEPTH_TEST); //to use the depth test
	glDepthMask(GL_FALSE); //But to not modify the depth buffer by any means
	glDepthFunc(GL_GEQUAL); //But only render if the object is inside the depth
	glEnable(GL_CULL_FACE); //And render the inner side of the cube
	glFrontFace(GL_CW);

	

	//draw again inside the gbuffers
	gbuffers->bind();
	camera->enable();

	GFX::Shader* decal_shader = GFX::Shader::Get("decals");
	assert(decal_shader);

	decal_shader->enable();

	Matrix44 model;
	model.setTranslation(-28, 53, -47);
	model.scale(5, 0.5, 5);

	decal_shader->setUniform("u_model",model);
	decal_shader->setUniform("u_color", vec4(1.0, 0.0, 0.0, 1.0));

	decal_shader->setUniform("u_camera_pos", camera->eye);
	decal_shader->setUniform("u_viewprojection", camera->viewprojection_matrix);

	decal_shader->setUniform("u_depth_texture", cloned_depth_texture, 0);


	decal_shader->setUniform("u_inverse_viewprojection", camera->inverse_viewprojection_matrix);
	decal_shader->setUniform("u_iRes", vec2(1.0 / CORE::getWindowSize().x, 1.0 / CORE::getWindowSize().y));
;
	Matrix44 model_inverse = model;
	model_inverse.inverse();
	decal_shader->setUniform("u_inv_decal_model", model_inverse);

	GFX::Texture* decal_texture = GFX::Texture::Get("data/decals/graffitti.png", true, true);
	decal_shader->setUniform("u_decal_texture", decal_texture, 1);

	cube.render(GL_TRIANGLES);

	gbuffers->unbind();

	glColorMask(true, true, true, true);

	glDepthFunc(GL_LEQUAL);
	glFrontFace(GL_CCW);


	glDepthMask(GL_TRUE);
	glDisable(GL_CULL_FACE); */
	
	//*________________________________________________
	
	ssaoBlur(camera);

	//render illumination scene
	renderIlluminationScene(camera, &size);

	//render irradiance
	renderIrradianceScene(camera, &size);


	//FINAL FBO TO BE DISPLAYED ON THE SCREEN
	if (!combined_illumination_fbo)
	{
		combined_illumination_fbo = new GFX::FBO();
		combined_illumination_fbo->create(size.x, size.y, 1, GL_RGB, GL_FLOAT, false);
		combined_illumination_fbo->color_textures[0]->setName("total_illum");

	}

	combined_illumination_fbo->bind();
	GFX::Shader* combine_shader = GFX::Shader::Get("combine");
	combine_shader->enable();
	combine_shader->setUniform("u_illumination_texture", illumination_fbo->color_textures[0], 0);
	
	if(combined_irr)
		combine_shader->setUniform("u_probe_illumination_texture", probe_illumination_fbo->color_textures[0], 1);
	else
		combine_shader->setUniform("u_probe_illumination_texture", GFX::Texture().getBlackTexture(), 1);
	GFX::Mesh::getQuad()->render(GL_TRIANGLES);
	combine_shader->disable();


	if (!use_dithering) {
		sort(alpha_renderables.begin(), alpha_renderables.end(), renderableComparator);
		for (Renderable& re : alpha_renderables)
		{
			if (camera->testBoxInFrustum(re.bounding.center, re.bounding.halfsize))
				renderMeshWithMaterialLights(re.model, re.mesh, re.material);
		}
	}

	renderFog(camera);

	combined_illumination_fbo->unbind();
	
	if (!renderFBO)
	{
		renderFBO = new GFX::FBO();
		renderFBO->create(size.x, size.y, 1, GL_RGB, GL_FLOAT, false);
	}

	renderFBO->bind();

	if (use_degamma)
		illumination_fbo->color_textures[0]->toViewport(GFX::Shader::Get("gamma"));
	else if (combined_irr)
		combined_illumination_fbo->color_textures[0]->toViewport();
	else
		illumination_fbo->color_textures[0]->toViewport();

	if (use_volumetric)
	{
		if (volumetric_fbo)
		{
			glEnable(GL_DEPTH_TEST);
			glEnable(GL_BLEND);
			glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
			volumetric_fbo->color_textures[0]->toViewport();
			glDisable(GL_BLEND);
		}
	}

	renderFBO->unbind();

	/*if (!postfxIN_fbo)
	{
		postfxIN_fbo = new GFX::FBO();
		postfxOUT_fbo = new GFX::FBO();

		postfxIN_fbo->create(size.x, size.y, 1, GL_RGB, GL_HALF_FLOAT, false);
		postfxOUT_fbo->create(size.x, size.y, 1, GL_RGB, GL_HALF_FLOAT, false);
	}
	postfxIN_fbo->bind();
	renderFBO->color_textures[0]->toViewport();
	postfxIN_fbo->unbind();

	bloom();*/

	if (!postfx_fbo)
	{
		postfx_fbo = new GFX::FBO();
		postfx_fbo->create(size.x, size.y, 1, GL_RGB, GL_HALF_FLOAT, false);
	}

	postfx_fbo->bind();
	renderFBO->color_textures[0]->toViewport();
	postfx_fbo->unbind();

	postfx_fbo->bind();
	if (use_simpleblurr) simpleBlur();
	postfx_fbo->unbind();

	postfx_fbo->bind();
	if (use_dof) postDepthOfField(camera);
	postfx_fbo->unbind();

	postfx_fbo->bind();
	GFX::Shader* lut_shader;
	GFX::Shader* lut_shader2;
	if (use_lut) {
		lut_shader = GFX::Shader::Get("lut");
		lut_shader->enable();
		lut_shader->setUniform("u_textureB", GFX::Texture::Get("data/textures/brdfLUT.png", true, true), 1);
		lut_shader->setUniform("u_amount", lut_amount);
		postfx_fbo->color_textures[0]->toViewport(lut_shader);
	}
	else if (use_lut2) {
		lut_shader2 = GFX::Shader::Get("lut2");
		lut_shader2->enable();
		lut_shader2->setUniform("u_textureB", GFX::Texture::Get("data/textures/brdfLUT.png", true, true), 1);
		lut_shader2->setUniform("u_amount", lut_amount2);
		postfx_fbo->color_textures[0]->toViewport(lut_shader2);

	}
	postfx_fbo->unbind();

	postfx_fbo->bind();
	GFX::Shader* postfx_shader = GFX::Shader::Get("postfx");
	postfx_shader->enable();

	postfx_shader->setUniform("use_vignetting", use_vignetting);
	postfx_shader->setUniform("vignetting", vignetting);
	postfx_shader->setUniform("use_color_correction", use_color_correction);
	postfx_shader->setUniform("contrast", contrast);
	postfx_shader->setUniform("use_fish_eye", use_fish_eye);
	postfx_shader->setUniform("u_fish_eye_strength", fish_eye_strength);
	postfx_shader->setUniform("use_chromatic_aberration", use_chromatic_aberration);
	postfx_shader->setUniform("ca_strength", 15.0f);
	postfx_shader->setUniform("iRes", vec2(1.0f/size.x, 1.0f/size.y));
	postfx_shader->setUniform("Res", size);
	postfx_fbo->color_textures[0]->toViewport(postfx_shader);
	postfx_fbo->unbind();


	if (use_tonemapper) renderTonemapper(postfx_fbo->color_textures[0]);
	else postfx_fbo->color_textures[0]->toViewport();

	glDepthFunc(GL_LESS);
	glEnable(GL_DEPTH_TEST);

	if (show_probes) visualizeGrid(2);
	if (render_refelction_probes) visualizeReflectionProbes(10.0f);

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
	if (view_blur)
		blur_fbo->color_textures[0]->toViewport();
}

void Renderer::bloom()
{
	//four iterations blur
	vec2 size = CORE::getWindowSize();
	float width = size.x, height = size.y;
	GFX::Shader* shader = GFX::Shader::Get("bloom");
	shader->enable();

	int power = 1;
	for (int i = 0; i < 4; ++i)
	{
		//horizontal blur
		shader->setUniform("u_offset", vec2(1.0f / width, 0.0) * (float)power);
		shader->setUniform("u_texture", postfxIN_fbo->color_textures[0], 0);
		postfxOUT_fbo->bind();
		GFX::Mesh::getQuad()->render(GL_TRIANGLES);
		postfxOUT_fbo->unbind();
		std::swap(postfxIN_fbo, postfxOUT_fbo);

		//vertical blur
		shader->setUniform("u_offset", vec2(0.0, 1.0f / height) * (float)power);
		shader->setUniform("u_texture", postfxIN_fbo->color_textures[0], 0);
		postfxOUT_fbo->bind();
		GFX::Mesh::getQuad()->render(GL_TRIANGLES);
		postfxOUT_fbo->unbind();
		std::swap(postfxIN_fbo, postfxOUT_fbo);

		power = power << 1;
	}

}

void Renderer::simpleBlur()
{
	vec2 size = CORE::getWindowSize();
	GFX::Mesh* quad = GFX::Mesh::getQuad();

	GFX::Shader* shader = GFX::Shader::Get("simpleBlur");
	shader->enable();

	shader->setUniform("u_texture", postfx_fbo->color_textures[0], 0);
	shader->setUniform("iRes", vec2(1.0f / size.x, 1.0f / size.y));
	shader->setUniform("u_scale", df_scale_blur);

	quad->render(GL_TRIANGLES);
}

void Renderer::postDepthOfField(Camera* camera)
{
	vec2 size = CORE::getWindowSize();
	GFX::Mesh* quad = GFX::Mesh::getQuad();

	GFX::Shader* dof_shader = GFX::Shader::Get("depthoffield");
	assert(dof_shader);

	dof_shader->enable();

	// Set uniforms for depth of field shader
	dof_shader->setUniform("u_focus_texture", renderFBO->color_textures[0], 0);
	dof_shader->setUniform("u_unfocus_texture", postfx_fbo->color_textures[0], 1);

	dof_shader->setUniform("u_depth_texture", gbuffers->depth_texture, 2);

	// Depth of field parameters
	dof_shader->setUniform("u_min_distance", df_min_distance);
	dof_shader->setUniform("u_max_distance", df_max_distance);
	dof_shader->setUniform("u_focal_distance", df_focal_distance);
	dof_shader->setUniform("u_scale_blur", df_scale_blur);

	dof_shader->setUniform("camera_near", camera->near_plane);
	dof_shader->setUniform("camera_far", camera->far_plane);

	dof_shader->setUniform("u_iRes", vec2(1.0 / size.x, 1.0 / size.y));

	// Render the quad
	quad->render(GL_TRIANGLES);
	dof_shader->disable();
}


void Renderer::renderDecals(SCN::Scene* scene, Camera* camera, GFX::FBO* gbuffers)
{
	//enable alpha blending
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

	//block changing alpha channel
	glColorMask(true, true, true, false);

	glEnable(GL_DEPTH_TEST); //to use the depth test
	glDepthMask(GL_FALSE); //But to not modify the depth buffer by any means
	glDepthFunc(GL_GEQUAL); //But only render if the object is inside the depth
	glEnable(GL_CULL_FACE); //And render the inner side of the cube
	glFrontFace(GL_CW);

	//block from writing to normalmap gbuffer
	gbuffers->enableBuffers(false, true, false, false); //normalmap our second pos

	if (!cloned_depth_texture)
		cloned_depth_texture = new GFX::Texture();


	//copy gbuffers depth to other texture
	gbuffers->depth_texture->copyTo(cloned_depth_texture);

	//draw again inside the gbuffers
	gbuffers->bind();

	GFX::Shader* decal_shader = GFX::Shader::Get("decals");
	assert(decal_shader);

	decal_shader->enable();


	Matrix44 model;
	model.setTranslation(12, 126, 26);
	model.scale(5, 5, 5);


	decal_shader->setUniform("u_model", model);
	decal_shader->setUniform("u_color", vec4(1.0, 0.0, 0.0, 1.0));


	cameraToShader(camera, decal_shader);

	//cube.render(GL_TRIANGLES);

	//decal_shader->disable();

	/**for (auto decal : decals)
	{
		GFX::Shader* decal_shader = GFX::Shader::Get("decals");
		assert(decal_shader);

		decal_shader->enable();

		decal_shader->setUniform("u_depth_texture", cloned_depth_texture,0);


		decal_shader->setUniform("u_inverse_viewprojection", camera->inverse_viewprojection_matrix);
		decal_shader->setUniform("u_iRes", vec2(1.0 / CORE::getWindowSize().x, 1.0 / CORE::getWindowSize().y));

		//draw cube per decal


		Matrix44 model;
		model.setTranslation(-25,53,-52);
		model.scale(3, 3, 3);



		decal_shader->setUniform("u_model", model);
		decal_shader->setUniform("u_inv_decal_model", model.inverse());

		GFX::Texture* decal_texture = GFX::Texture::Get("data/decals/graffitti.png", true, true);
		decal_shader->setUniform("u_decal_texture", decal_texture);


		//decal_shader->setUniform("u_texture", decal->texture);


		cube.render(GL_TRIANGLES);

		decal_shader->disable();
	}*/

	//gbuffers->unbind();

	glColorMask(true, true, true, true);

	glDepthFunc(GL_LEQUAL);
	glFrontFace(GL_CCW);


	glDepthMask(GL_TRUE);
	glDisable(GL_CULL_FACE);


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
		BoundingBox world_bounding = transformBoundingBox(node_model, node->mesh->box);

		//if bounding box is inside the camera frustum then the object is probably visible
		if (camera->testBoxInFrustum(world_bounding.center, world_bounding.halfsize))
		{
			if (render_boundaries)
				node->mesh->renderBounding(node_model, true);
			renderMeshWithMaterial(node_model, node->mesh, node->material);
		}
	}

	//iterate recursively with children
	for (int i = 0; i < node->children.size(); ++i)
		renderNode(node->children[i], camera);
}
void  SCN::Renderer::captureProbes() {

	for (auto& probe : probes) {
		captureProbe(probe);  
	}

	if (probes_texture)
		delete probes_texture;


	//create the texture to store the probes (do this ONCE!!!)
	probes_texture = new GFX::Texture(
		9, //9 coefficients per probe
		(int)probes.size(), //as many rows as probes
		GL_RGB, //3 channels per coefficient
		GL_FLOAT); //they require a high range
	//we must create the color information for the texture. because every SH are 27 floats in the RGB,RGB,... order, we can create an array of SphericalHarmonics and use it as pixels of the texture

	SphericalHarmonics* sh_data = NULL;
	sh_data = new SphericalHarmonics[probes_info.dim.x * probes_info.dim.y * probes_info.dim.z];

	//here we fill the data of the array with our probes in x,y,z order
	for (int i = 0; i < probes.size(); ++i)
		sh_data[i] = probes[i].sh; //una probe por row

	//now upload the data to the GPU as a texture
	probes_texture->upload(GL_RGB, GL_FLOAT, false, (uint8*)sh_data);

	//disable any texture filtering when reading
	probes_texture->bind();
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);

	//always free memory after allocating it!!!
	delete[] sh_data;

}

void SCN::Renderer::captureProbe(sProbe& p) //cuanta luz llega a esa esfera
{
	//poner camara donde esta la probe y tira renderer en 6 direcciones

	static FloatImage images[6]; //here we will store the six views  //static reusa la var
	if (!irr_fbo) {

		irr_fbo = new GFX::FBO();
		irr_fbo->create(64, 64, 1, GL_RGB, GL_FLOAT, false);
	}

	Camera cam;

	//set the fov to 90 and the aspect to 1 (cuadrados)
	cam.setPerspective(90, 1, 0.1, 1000);

	for (int i = 0; i < 6; ++i) //for every cubemap face
	{
		//compute camera orientation using defined vectors
		vec3 eye = p.pos;
		vec3 front = cubemapFaceNormals[i][2];
		vec3 center = p.pos + front;
		vec3 up = cubemapFaceNormals[i][1];
		cam.lookAt(eye, center, up);
		cam.enable();

		//render the scene from this point of view
		irr_fbo->bind();

		renderSceneForward(scene, &cam);

		irr_fbo->unbind();

		//read the pixels back and store in a FloatImage
		images[i].fromTexture(irr_fbo->color_textures[0]); //copia de GPU a CPU
	}

	//compute the coefficients given the six images
	p.sh = computeSH(images); //opicon gamma o degamma en la funcion si queremos incluirlo!


}

void Renderer::renderFog(Camera* camera) 
{
	vec2 size = CORE::getWindowSize();
	int shadowMapPos = 8;

	if (!volumetric_fbo)
	{
		volumetric_fbo = new GFX::FBO();
		volumetric_fbo->create(size.x / 2, size.y / 2, 3, GL_RGBA, GL_UNSIGNED_BYTE, false);
	}

	volumetric_fbo->bind();

	glClear(GL_COLOR_BUFFER_BIT);

	glDisable(GL_DEPTH_TEST);
	glDisable(GL_BLEND);

	GFX::Shader* shader = GFX::Shader::Get("volumetric");
	shader->enable();
	shader->setTexture("u_depth_texture", gbuffers->depth_texture, 0);
	shader->setMatrix44("u_inverse_viewprojection", camera->inverse_viewprojection_matrix);
	shader->setUniform("u_iRes", vec2(1.0 / volumetric_fbo->color_textures[0]->width, 1.0 / volumetric_fbo->color_textures[0]->height));

	cameraToShader(camera, shader);
	shader->setUniform("u_air_density", air_density);
	shader->setUniform("u_ambient_light", scene->ambient_light);
	shader->setUniform("u_constant_density", (float)constant_density);
	shader->setUniform("u_time", (float)getTime() * 0.001f);

	GFX::Mesh* quad = GFX::Mesh::getQuad();

	if (!mainLight) return;
	lightToShader(mainLight, shader);
	if (mainLight->cast_shadows && mainLight->shadowMapFBO)
		{
			shadowToShader(mainLight, shadowMapPos, shader);
		}
		else
			shader->setUniform("u_light_cast_shadows", 0);
	quad->render(GL_TRIANGLES);

	//for (LightEntity* light : lights)
	//{
	//	if (light->light_type == POINT)
	//		continue;

	//	//upload uniforms and render
	//	lightToShader(light, shader);

	//	if (light->cast_shadows && light->shadowMapFBO)
	//	{
	//		shadowToShader(light, shadowMapPos, shader);
	//	}
	//	else
	//		shader->setUniform("u_light_cast_shadows", 0);

	//	//only upload once
	//	shader->setUniform("u_ambient_light", vec3(0.0f));
	//	glEnable(GL_BLEND);
	//	quad->render(GL_TRIANGLES);
	//}
	//quad->render(GL_TRIANGLES);

	glEnable(GL_DEPTH_TEST);
	glDisable(GL_BLEND);

	volumetric_fbo->unbind();
}

void  SCN::Renderer::visualizeGrid(float scale) {

	for (auto& probe : probes) {
		renderProbe(probe.pos, scale, probe.sh);
	}
}
void  SCN::Renderer::renderProbe(vec3 pos, float scale, SphericalHarmonics& shs) {

	Camera* camera = Camera::current;

	glDisable(GL_BLEND);
	glEnable(GL_CULL_FACE); //para no pintar parte interior


	GFX::Shader* shader = GFX::Shader::Get("probe");
	if (!shader)
		return;
	shader->enable();

	Matrix44 m;
	m.setTranslation(pos.x, pos.y, pos.z);
	m.scale(scale, scale, scale);
	shader->setUniform("u_model", m);
	cameraToShader(camera, shader);
	shader->setUniform3Array("u_coefs", shs.coeffs[0].v, 9);
	sphere.render(GL_TRIANGLES);
	shader->disable();
}
void SCN::Renderer::captureReflectionProbes() {

	if (!reflections_fbo)
	{
		vec2 size = CORE::getWindowSize();
		reflections_fbo = new GFX::FBO();
		reflections_fbo->create(size.x, size.y, 1, GL_RGB, GL_FLOAT, false);
	}

	for (auto& p : reflection_probes)
		captureReflectionProbe(p);
}

void SCN::Renderer::captureReflectionProbe(sReflectionProbe& p) {

	if (!p.cubemap)
	{
		//espacio para el cubemap
		p.cubemap = new GFX::Texture();
		p.cubemap->createCubemap(512, 512, NULL, GL_RGB, GL_UNSIGNED_INT, true);	//mipmaps //true->Minamps

	}

	Camera cam;

	//set the fov to 90 and the aspect to 1 
	cam.setPerspective(90, 1, 0.1, 1000);

	for (int i = 0; i < 6; ++i) //for every cubemap face
	{
		reflections_fbo->setTexture(p.cubemap, i);

		//bind FBO
		reflections_fbo->bind();

		//compute camera orientation using defined vectors
		vec3 eye = p.pos;
		vec3 center = p.pos + cubemapFaceNormals[i][2];
		vec3 up = cubemapFaceNormals[i][1];
		cam.lookAt(eye, center, up);
		cam.enable();
		renderSceneForward(scene, &cam);
		reflections_fbo->unbind();


	}

	//generate the mipmaps
	glEnable(GL_TEXTURE_CUBE_MAP_SEAMLESS);
	p.cubemap->generateMipmaps();


}

void SCN::Renderer::visualizeReflectionProbes(float scale) {

	glDisable(GL_BLEND);
	glEnable(GL_CULL_FACE); //para no pintar parte interior
	for (auto& ref : reflection_probes) {
		renderReflectionProbe(ref, scale);
	}
}
void SCN::Renderer::renderReflectionProbe(sReflectionProbe& p, float scale) {
	

	Camera* camera = Camera::current;

	GFX::Shader* sh_reflc = GFX::Shader::Get("reflecionProbe");
	assert(sh_reflc);

	sh_reflc->enable();

	GFX::Texture* texture = p.cubemap ? p.cubemap : skybox_cubemap;

	if (!texture)
		return;

	Matrix44 m;
	m.setTranslation(p.pos.x, p.pos.y, p.pos.z);
	m.scale(scale, scale, scale);
	sh_reflc->setUniform("u_model", m);
	cameraToShader(camera, sh_reflc);

	sh_reflc->setUniform("u_environment_texture", texture, 0); //reflections texture
	sh_reflc->setUniform("u_metallic_roughness_texture", gbuffers->color_textures[3], 1);
	sh_reflc->setUniform("u_color_texture", gbuffers->color_textures[0], 2);
	sh_reflc->setUniform("u_camera_pos", camera->eye);


	vec2 size = CORE::getWindowSize();
	sh_reflc->setUniform("u_iRes", vec2(1.0 / size.x, 1.0 / size.y));
	sphere.render(GL_TRIANGLES);
	sh_reflc->disable();

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

	//added---------------------------------------------------------------

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
		shader->setUniform("normal_option", (int)use_normal_map);
	}

	shader->setUniform("occlusion_option", (int)use_occlusion);
	shader->setUniform("emissive_option", (int)use_emissive);
	if (use_dithering && material->alpha_mode == eAlphaMode::BLEND)
		shader->setUniform("dithering_option", 1);
	else
		shader->setUniform("dithering_option", 0);


	//---------------------------------------------------------------------


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
	if (!mesh || !mesh->getNumVertices() || !material)
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
	float t = getTime();
	shader->setUniform("u_time", t);

	shader->setUniform("u_color", material->color);
	if (texture)
		shader->setUniform("u_texture", texture, 0);

	//this is used to say which is the alpha threshold to what we should not paint a pixel on the screen (to cut polygons according to texture alpha)
	shader->setUniform("u_alpha_cutoff", material->alpha_mode == SCN::eAlphaMode::MASK ? material->alpha_cutoff : 0.001f);

	if (render_wireframe)
		glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);

	//do the draw call that renders the mesh into the screen
	mesh->render(GL_TRIANGLES);

	//disable shader
	shader->disable();

	//set the render state as it was before to avoid problems with future renders
	glDisable(GL_BLEND);
	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
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
	if (!mesh || !mesh->getNumVertices() || !material)
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
	if (material->two_sided)
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
	shader->setUniform("u_time", t);

	//ambient light
	shader->setUniform("u_ambient_light", scene->ambient_light);

	//uniforms to calculate specular: camera position, alpha shininess, specular factor
	shader->setUniform("eye", camera->eye);
	shader->setUniform("u_alpha", material->roughness_factor);
	float specular_factor = material->metallic_factor;
	shader->setUniform("u_specular", material->metallic_factor);
	shader->setUniform("specular_option", use_specular && specular_factor > 0.0f);

	//color
	shader->setUniform("u_color", material->color);
	if (texture)
		shader->setUniform("u_texture", texture, texPosition++);

	//upload of normal, emissive and occlusion (red value of metallic roughness as specified) textures, if they don't exist upload white texture
	GFX::Texture* normalMap = material->textures[eTextureChannel::NORMALMAP].texture;
	GFX::Texture* emissiveTex = material->textures[eTextureChannel::EMISSIVE].texture;
	GFX::Texture* occlusionTex = material->textures[eTextureChannel::METALLIC_ROUGHNESS].texture;

	emissiveTex = (emissiveTex) ? emissiveTex : GFX::Texture::getWhiteTexture();
	occlusionTex = (occlusionTex) ? occlusionTex : GFX::Texture::getWhiteTexture();

	if (!normalMap)
		shader->setUniform("normal_option", (int)0);
	else {
		shader->setUniform("u_normal_texture", normalMap, texPosition++);
		shader->setUniform("normal_option", (int)use_normal_map);
	}

	if (illumination_fbo->color_textures[0] && (int)pipeline_mode == ePipelineMode::DEFERRED)
		shader->setUniform("u_depth_texture", illumination_fbo->color_textures[0], texPosition++);


	shader->setUniform("u_emissive_factor", material->emissive_factor);
	shader->setUniform("u_emissive_texture", emissiveTex, texPosition++);
	shader->setUniform("emissive_option", (int)use_emissive);

	shader->setUniform("u_metallic_roughness_texture", occlusionTex, texPosition++);
	shader->setUniform("occlusion_option", (int)use_occlusion);
	shader->setUniform("u_deferred_option", (int)pipeline_mode == ePipelineMode::DEFERRED);

	//this is used to say which is the alpha threshold to what we should not paint a pixel on the screen (to cut polygons according to texture alpha)
	shader->setUniform("u_alpha_cutoff", material->alpha_mode == SCN::eAlphaMode::MASK ? material->alpha_cutoff : 0.001f);

	if (render_wireframe)
		glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);

	if (lights.size()) {
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
				shader->setUniform("u_ambient_light", vec3(0.0f));
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
	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
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

	shader->setUniform1Array("u_light_cast_shadows", (int*)&u_light_cast_shadows_arr, N_LIGHTS);
	shader->setUniform1Array("u_shadowmap_bias", (float*)&u_shadowmap_biases, N_LIGHTS);
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
	vec2 cone_info = vec2(cosf(light->cone_info.x * PI / 180.0f), cosf(light->cone_info.y * PI / 180.0f));

	shader->setUniform("u_light_type", (int)light->light_type);
	shader->setUniform("u_light_position", light->root.model.getTranslation());
	shader->setUniform("u_light_front", light->root.model.frontVector().normalize());
	shader->setUniform("u_light_color", light->color * light->intensity);
	shader->setUniform("u_light_max_distance", light->max_distance);

	if (use_degamma) {

		vec3 fcolor = light->color * light->intensity;
		vec3 correctedColor = vec3(pow(fcolor.x, 2.2), pow(fcolor.y, 2.2), pow(fcolor.z, 2.2));
		shader->setUniform("u_light_color", correctedColor);
	}
	else
		shader->setUniform("u_light_color", light->color * light->intensity);

	shader->setUniform("u_light_max_distance", light->max_distance);
	shader->setUniform("u_light_cone_info", cone_info);
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

		vec2 cone_info = vec2(cosf(light->cone_info.x * PI / 180.0f), cosf(light->cone_info.y * PI / 180.0f));

		light_position[i] = light->getGlobalPosition();
		light_color[i] = light->color * lights[i]->intensity;
		light_types[i] = (int)light->light_type;
		light_max_distances[i] = light->max_distance;
		cone_infos[i] = cone_info;
		light_fronts[i] = light->root.model.frontVector().normalize();
	}

	shader->setUniform3Array("u_light_pos", (float*)&light_position, N_LIGHTS);
	shader->setUniform3Array("u_light_color", (float*)&light_color, N_LIGHTS);
	shader->setUniform1Array("u_light_type", (int*)&light_types, N_LIGHTS);
	shader->setUniform1Array("u_light_max_distance", (float*)&light_max_distances, N_LIGHTS);
	shader->setUniform2Array("u_light_cone_info", (float*)&cone_infos, N_LIGHTS);
	shader->setUniform3Array("u_light_front", (float*)&light_fronts, N_LIGHTS);
	shader->setUniform1("u_num_lights", N_LIGHTS);
}

void SCN::Renderer::cameraToShader(Camera* camera, GFX::Shader* shader)
{
	shader->setUniform("u_viewprojection", camera->viewprojection_matrix);
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
	ImGui::Checkbox("Dithering", &use_dithering);

	if (ImGui::TreeNode("SSAO+BLUR"))
	{
		ImGui::Checkbox("Use SSAO", &use_ssao);
		ImGui::Checkbox("Use blur", &use_blur);

		if (use_ssao || use_blur) {
			ImGui::DragFloat("ssao radius", &ssao_radius, 0.01f, 0.0f);
			ImGui::DragFloat("ssao max distance", &ssao_max_distance, 0.01f, 0.0f);

			ImGui::Checkbox("View ssao", &view_ssao);
			ImGui::Checkbox("View blur", &view_blur);
		}
		ImGui::TreePop();
	}

	ImGui::Checkbox("Use degamma", &use_degamma);


	if (ImGui::TreeNode("Irradiance"))
	{
		ImGui::Checkbox("Show irradiance probes", &show_probes);
		ImGui::Checkbox("Scene with irradiance", &combined_irr);
		if (ImGui::Button("Capture Irradiance"))
			captureProbes();
	
		ImGui::TreePop();
	}


	if (ImGui::TreeNode("Relfections"))
	{
		ImGui::Checkbox("Show reflection probes", &render_refelction_probes);
		if (ImGui::Button("Capture Refl.probes"))
			captureReflectionProbes();

		ImGui::TreePop();
	}

	if (ImGui::TreeNode("postfx"))
	{
		ImGui::Checkbox("Simple Blurr", &use_simpleblurr);

		//postDepthofField
		ImGui::Checkbox("Depth of field", &use_dof);
		if (use_dof) {
			use_simpleblurr = true;
			ImGui::DragFloat("Dof min distance", &df_min_distance, 0.01f, 0.0f, df_max_distance);
			ImGui::DragFloat("DoF max distance", &df_max_distance, 0.01f, df_min_distance, 100000.0f);
			ImGui::DragFloat("DoF focal distance", &df_focal_distance, 0.01f, 0.01f, 100000.0f);
		}
		if (use_dof || use_simpleblurr)
			ImGui::DragFloat("Blur scale", &df_scale_blur, 0.01f, 0, 100);

		ImGui::Checkbox("Color Correction", &use_color_correction);
		if (use_color_correction)
		{
			ImGui::DragFloat("Contrast Red", &contrast.x, 0.01f, 0, 100);
			ImGui::DragFloat("Contrast Green", &contrast.y, 0.01f, 0, 100);
			ImGui::DragFloat("Contrast Blue", &contrast.z, 0.01f, 0, 100);
		}

		ImGui::Checkbox("Vignetting", &use_vignetting);
		if (use_vignetting)
			ImGui::DragFloat("Vignetting", &vignetting, 0.01f, 0, 2);

		ImGui::Checkbox("Fish eye", &use_fish_eye);
		if (use_fish_eye)
			ImGui::DragFloat("Fish eye strength", &fish_eye_strength, 0.01f, 0.0f, 1.0f, "%0.3f");

		ImGui::Checkbox("Chromatic aberration", &use_chromatic_aberration);
		if (use_chromatic_aberration)
			ImGui::DragFloat("CA Strength", &ca_strength, 0.01f, 0.0f, 100.0f, "%0.3f");

		ImGui::Checkbox("lut", &use_lut);
		if (use_lut)
			ImGui::DragFloat("lut amount", &lut_amount, 0.01f, 0.0f, 100.0f, "%0.3f");

		ImGui::Checkbox("lut2", &use_lut2);
		if (use_lut2)
			ImGui::DragFloat("lut amount2", &lut_amount2, 0.01f, 0.0f, 100.0f, "%0.3f");
		ImGui::TreePop();

		ImGui::TreePop();
	}

	//effects post
	//ImGui::Combo("Show GBuffer", (int*)&post_fx, "NONE\0DEPTHOFFIELD\0BLOOM\0POST_COUNT\0", ePost_fx::POST_COUNT);
	
	if (ImGui::TreeNode("Tonemapper Parameters"))
	{
		ImGui::Checkbox("Use Tonemapper", &use_tonemapper);
		if (use_tonemapper)
		{
			ImGui::DragFloat("Scale", &tonemapper_scale, 0.01f, 0.0f, 2.0f);
			ImGui::DragFloat("Average Lum", &tonemapper_avg_lum, 0.01f, 0.0f, 2.0f);
			ImGui::DragFloat("Lum White", &tonemapper_lumwhite, 0.01f, 0.0f, 2.0f);
		}
		ImGui::TreePop();
	}

	if (ImGui::TreeNode("Volumetric"))
	{
		ImGui::Checkbox("Use volumetrics", &use_volumetric);

		if (use_volumetric) {
			ImGui::DragFloat("Air density", &air_density, 0.00001f, 0.0f, 1.0f, "%0.5f");
			ImGui::Checkbox("Use constant density", &constant_density);
		}
		ImGui::TreePop();
	}

	if (ImGui::Button("ShadowMap 256"))
		shadowmap_size = 256;
	if (ImGui::Button("ShadowMap 512"))
		shadowmap_size = 512;
	if (ImGui::Button("ShadowMap 1024"))
		shadowmap_size = 1024;
	if (ImGui::Button("ShadowMap 2048"))
		shadowmap_size = 2048;
}

void Renderer::resize()
{
	vec2 size = CORE::getWindowSize();

	if (!gbuffers)
		gbuffers = new GFX::FBO();

	gbuffers->create(size.x, size.y, 4, GL_RGBA, GL_UNSIGNED_BYTE, true);  //crea todas las texturas attached, true if we want depthbuffer in a texure (para poder leerlo)

	if (!illumination_fbo)
		illumination_fbo = new GFX::FBO();

	illumination_fbo->create(size.x, size.y, 1, GL_RGB, GL_FLOAT, false);

	if (!combined_illumination_fbo)
	{
		combined_illumination_fbo = new GFX::FBO();
		combined_illumination_fbo->color_textures[0]->setName("total_illum");
	}

	combined_illumination_fbo->create(size.x, size.y, 1, GL_RGB, GL_FLOAT, false);

	if (!probe_illumination_fbo)
	{
		probe_illumination_fbo = new GFX::FBO();
		probe_illumination_fbo->color_textures[0]->setName("irradiance_probe");
	}
	probe_illumination_fbo->create(size.x, size.y, 1, GL_RGB, GL_FLOAT, false);

	if (!volumetric_fbo)
		volumetric_fbo = new GFX::FBO();
	
	volumetric_fbo->create(size.x / 2, size.y / 2, 3, GL_RGBA, GL_UNSIGNED_BYTE, false);

	if (!renderFBO)
		renderFBO = new GFX::FBO();
	renderFBO->create(size.x, size.y, 1, GL_RGB, GL_FLOAT, false);

	if (!blur_fbo)
		blur_fbo = new GFX::FBO();
	blur_fbo->create(size.x, size.y, 1, GL_RGB, GL_HALF_FLOAT, false);
}

#else
void Renderer::showUI() {}
#endif