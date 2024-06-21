#pragma once
#include "scene.h"
#include "prefab.h"

#include "light.h"
#include "../gfx/sphericalharmonics.h"

#define MAX_SHADOWS 4

//forward declarations
class Camera;
class Skeleton;
namespace GFX {
	class Shader;
	class Mesh;
	class FBO;
}

enum ePipelineMode
{
	FLAT,
	FORWARD,
	DEFERRED,
	PIPELINECOUNT
};

enum eShowGBuffer
{
	NONE,
	COLOR,
	NORMAL,
	EXTRA,
	DEPTH,
	SHOW_BUFFER_COUNT
};

enum ePost_fx {
	DEPTHOFFIELD,
	BLOOM
};
struct sProbe {
	vec3 pos; //where is located
	vec3 local; //its ijk pos in the matrix
	int index; //its index in the linear array
	SphericalHarmonics sh; //coeffs
};

//struct to store grid info
struct sIrradianceInfo {
	vec3 start;
	vec3 end;
	vec3 dim;
	vec3 delta;
	int num_probes;
};

struct sReflectionProbe {
	vec3 pos;
	GFX::Texture* cubemap = NULL ;
};


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
		static Renderer* instance;

		bool render_wireframe;
		bool render_boundaries;
		bool use_multipass_lights;
		bool use_no_texture;
		bool use_normal_map;
		bool use_emissive;
		bool use_specular;
		bool use_occlusion;
		bool view_ssao;
		bool use_ssao;
		bool use_degamma;
		bool view_blur;
		bool use_blur;
		bool use_dithering;
		bool use_tonemapper;
		//volumetrics
		bool use_volumetric;
		bool constant_density;
		bool use_simpleblurr;
		float air_density;

		bool show_probes;
		bool render_refelction_probes;
		//temporal
		bool combined_irr;

		bool use_dof;
		//tonemapper
		float curr_tonemapper;
		float tonemapper_scale;
		float tonemapper_avg_lum;
		float tonemapper_lumwhite;

		float ssao_radius;
		float ssao_max_distance;

		//post depth of field
		float df_min_distance;
		float df_max_distance;
		float df_focal_distance;
		float df_scale_blur;

		std::vector<vec3> random_points;

		ePipelineMode pipeline_mode;
		eShowGBuffer gbuffer_show_mode;
		ePost_fx post_fx;

		int shadowmap_size;

		GFX::Mesh* light_sphere;
		GFX::Texture* skybox_cubemap;

		SCN::Scene* scene;

		//tmp containers
		std::vector<Renderable> renderables;
		std::vector<LightEntity*> lights;
		std::vector<Renderable> opaque_renderables;
		std::vector<Renderable> alpha_renderables;
		std::vector<LightEntity*> visibleLights;

		LightEntity* mainLight;

		//DecalEntity* decal;
		//std::vector<DecalEntity*> decals;

		void extractRenderables(SCN::Node* node, Camera* camera);

		void renderTonemapper(GFX::Texture* colorTex);

		//updated every frame
		Renderer(const char* shaders_atlas_filename);

		//just to be sure we have everything ready for the rendering
		void setupScene();

		//add here your functions
		void extractSceneInfo(SCN::Scene* scene, Camera* camera);
		static bool renderableComparator(const Renderable& a, const Renderable& b);
		//renders several elements of the scene
		void renderScene(SCN::Scene* scene, Camera* camera);
		void renderSceneForward(SCN::Scene* scene, Camera* camera);
		void renderSceneDeferred(SCN::Scene* scene, Camera* camera);
		void gbufferToShader(GFX::Shader* shader, vec2 size, Camera* camera);
		void lightsDeferred(Camera* camera);
		void renderIrradianceLights();
		void renderIrradianceScene(Camera* camera, vec2* size);
		void ssaoBlur(Camera* camera);
		void renderDecals(SCN::Scene* scene, Camera* camera, GFX::FBO* gbuffers);
		//render the skybox
		void renderSkybox(GFX::Texture* cubemap);

		void simpleBlur();
		void renderFog(Camera* camera);

		//to render one node from the prefab and its children
		void renderNode(SCN::Node* node, Camera* camera);

		void generateShadowMaps(Camera* camera);

		//to render one mesh given its material and transformation matrix
		void renderMeshWithMaterial(const Matrix44 model, GFX::Mesh* mesh, SCN::Material* material);

		void renderMeshWithMaterialPlain(const Matrix44 model, GFX::Mesh* mesh, SCN::Material* material);

		void renderMeshWithMaterialLights(const Matrix44 model, GFX::Mesh* mesh, SCN::Material* material);

		void renderMeshWithMaterialGBuffers(const Matrix44 model, GFX::Mesh* mesh, SCN::Material* material);

		//irradiance
		void captureProbe(sProbe& p); //rellena los spherical harmonics de la probe con la luz que llega al punto donde esta la probe
		void renderProbe(vec3 pos, float scale, SphericalHarmonics& shs);
		void visualizeGrid(float scale);
		void captureProbes();


		//reflection
		void captureReflectionProbe(sReflectionProbe& p);
		void renderReflectionProbe(sReflectionProbe& p,float scale);
		void visualizeReflectionProbes(float scale);
		void captureReflectionProbes();


		//post-procesing
		void postDepthOfField(Camera* camera);

		void showUI();

		void shadowToShader(GFX::Shader* shader);
		void shadowToShaderAuxiliar(GFX::Shader* shader);
		void shadowToShader(LightEntity* light, int& shadowMapPos, GFX::Shader* shader);
		void cameraToShader(Camera* camera, GFX::Shader* shader); //sends camera uniforms to shader
		void lightToShader(LightEntity* light, GFX::Shader* shader); //sends light uniforms to shader	
		void lightToShader(GFX::Shader* shader);
		void texturesToShader(SCN::Material* material, GFX::Shader* shader) const;

		void resize();
	};

};