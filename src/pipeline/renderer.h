#pragma once
#include "scene.h"
#include "prefab.h"

#include "light.h"

//forward declarations
class Camera;
class Skeleton;
namespace GFX {
	class Shader;
	class Mesh;
	class FBO;
}

namespace SCN {

	class Prefab;
	class Material;

	class Renderable
	{
	public:
		mat4 model;
		GFX::Mesh* mesh;
		SCN::Material* material;
		BoundingBox bounding;
		float dist_to_cam;
	};

	// This class is in charge of rendering anything in our system.
	// Separating the render from anything else makes the code cleaner
	class Renderer
	{
	public:
		bool render_wireframe;
		bool render_boundaries;
		bool use_multipass_lights;
		bool use_no_texture;
		bool use_normal_map;
		bool use_emissive;
		bool use_specular;
		bool use_occlusion;
		bool use_single_pass;

		GFX::Texture* skybox_cubemap;

		SCN::Scene* scene;

		//tmp containers
		std::vector<Renderable> renderables;
		std::vector<LightEntity*> lights;
		std::vector<LightEntity*> visibleLights;

		void extractRenderables(SCN::Node* node, Camera* camera);

		//updated every frame
		Renderer(const char* shaders_atlas_filename );

		//just to be sure we have everything ready for the rendering
		void setupScene();

		//add here your functions
		void extractSceneInfo(SCN::Scene* scene, Camera* camera);
		static bool renderableComparator(const Renderable& a, const Renderable& b);
		//renders several elements of the scene
		void renderScene(SCN::Scene* scene, Camera* camera);

		//render the skybox
		void renderSkybox(GFX::Texture* cubemap);
	
		//to render one node from the prefab and its children
		void renderNode(SCN::Node* node, Camera* camera);

		//to render one mesh given its material and transformation matrix
		void renderMeshWithMaterial(const Matrix44 model, GFX::Mesh* mesh, SCN::Material* material);

		void renderMeshWithMaterialLights(const Matrix44 model, GFX::Mesh* mesh, SCN::Material* material);

		void showUI();

		void cameraToShader(Camera* camera, GFX::Shader* shader); //sends camera uniforms to shader
		void lightToShader(LightEntity* light, GFX::Shader* shader); //sends light uniforms to shader	
	};

};