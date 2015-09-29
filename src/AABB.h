#ifndef __AABB__
#define __AABB__

#include "glm/glm.hpp"
#include "sceneStructs.h"

using namespace std;

struct AABB 
{
	glm::vec3 m_center; 
	glm::vec3 m_max_point;
	glm::vec3 m_min_point;
	glm::vec3 m_scale;

	Geom m_box;
	
	glm::vec3 getTranslation() { return m_center; }
	glm::vec3 getScale() { return m_scale; }

	

};



#endif /* defined(__Dress__AABB__) */
