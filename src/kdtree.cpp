#include <iostream>
#include <algorithm>
#include "scene.h"
#include "kdtree.h"

//this code only runs on cpu
AABB getAABB(const Geom & geom)
{
	AABB aabb;
	switch (geom.type)
	{
	case CUBE:
	{
		//vs 2012 way of init
		glm::vec4 tmp_arys[] = {
			glm::vec4(0.5f, 0.5f, 0.5f, 1.0f)
			, glm::vec4(0.5f, 0.5f, -0.5f, 1.0f)
			, glm::vec4(0.5f, -0.5f, 0.5f, 1.0f)
			, glm::vec4(-0.5f, 0.5f, 0.5f, 1.0f)
			, glm::vec4(0.5f, -0.5f, -0.5f, 1.0f)
			, glm::vec4(-0.5f, 0.5f, -0.5f, 1.0f)
			, glm::vec4(-0.5f, -0.5f, 0.5f, 1.0f)
			, glm::vec4(-0.5f, -0.5f, -0.5f, 1.0f)
		};
		std::vector<glm::vec4> points(&tmp_arys[0], &tmp_arys[0] + 8);

		glm::vec4 & t = points.at(0);
		aabb.max_pos = glm::vec3(-FLT_MAX, -FLT_MAX, -FLT_MAX);
		aabb.min_pos = glm::vec3(FLT_MAX, FLT_MAX, FLT_MAX);
		for (auto p : points)
		{
			p = geom.transform * p;
			p /= p.w;

			aabb.min_pos.x = min(aabb.min_pos.x, p.x);
			aabb.min_pos.y = min(aabb.min_pos.y, p.y);
			aabb.min_pos.z = min(aabb.min_pos.z, p.z);

			aabb.max_pos.x = max(aabb.max_pos.x, p.x);
			aabb.max_pos.y = max(aabb.max_pos.y, p.y);
			aabb.max_pos.z = max(aabb.max_pos.z, p.z);
		}
	}
	break;
	case SPHERE:
	{
		//simple square like cube
		//use max radius
		glm::vec4 tmp = geom.transform * glm::vec4(0.0f, 0.0f, 0.0f, 1.0f);
		glm::vec3 o(tmp.x / tmp.w, tmp.y / tmp.w, tmp.z / tmp.w);
		float r = max(geom.scale.x, geom.scale.y);
		r = 0.5f * max(r, geom.scale.z);
		glm::vec3 offset(r, r, r);
		aabb.min_pos = o - offset;
		aabb.max_pos = o + offset;
	}
	break;
	case TRIANGLE:
	{
		//ugly implementation
		const glm::vec3 & a = geom.translation;
		const glm::vec3 & b = geom.rotation;
		const glm::vec3 & c = geom.scale;

		float minx = a.x;
		float miny = a.y;
		float minz = a.z;

		float maxx = a.x;
		float maxy = a.y;
		float maxz = a.z;


		minx = min(minx, b.x);
		miny = min(miny, b.y);
		minz = min(minz, b.z);
		minx = min(minx, c.x);
		miny = min(miny, c.y);
		minz = min(minz, c.z);

		maxx = max(maxx, b.x);
		maxy = max(maxy, b.y);
		maxz = max(maxz, b.z);
		maxx = max(maxx, c.x);
		maxy = max(maxy, c.y);
		maxz = max(maxz, c.z);


		aabb.min_pos = glm::vec3(minx, miny, minz);
		aabb.max_pos = glm::vec3(maxx, maxy, maxz);
	}
	break;
	default:
		std::cerr << "GEOM TYPE ERROR\n";
		break;
	}
	return aabb;
}



std::pair<AABB, AABB> cutAABB(const AABB & parent, const AAPlane& pl)
{
	AABB l = parent;
	AABB r = parent;

	//suppose pl is always inside the parent aabb

	l.max_pos[pl.axis] = pl.pos;
	r.min_pos[pl.axis] = pl.pos;
	return std::make_pair(l, r);
}



typedef bool(*KdConstructCompareFun)(const KDNodeConstructWrapper &, const KDNodeConstructWrapper &);
bool my_kd_construct_compare_x(const KDNodeConstructWrapper & a, const KDNodeConstructWrapper & b)
{
	return a.mid.x < b.mid.x;
}
bool my_kd_construct_compare_y(const KDNodeConstructWrapper & a, const KDNodeConstructWrapper & b)
{
	return a.mid.y < b.mid.y;
}
bool my_kd_construct_compare_z(const KDNodeConstructWrapper & a, const KDNodeConstructWrapper & b)
{
	return a.mid.z < b.mid.z;
}

void KDTree::init(Scene & s)
{
	vector<Geom> & geoms_using = s.tmp_geoms;
	vector<Geom> & geoms_final = s.geoms;

	last_idx = 0;
	AABB spaceAABB;
	spaceAABB = getAABB(geoms_using.at(0));

	vector<KDNodeConstructWrapper> vec_geoms(geoms_using.size());

	int i = 0;
	for (const Geom & g : geoms_using)
	{

		vec_geoms.at(i).aabb = getAABB(g);
		vec_geoms.at(i).geom_idx = i;

		AABB & aabb = vec_geoms.at(i).aabb;

		vec_geoms.at(i).mid = (aabb.max_pos + aabb.min_pos) * 0.5f;

		//update spaceAABB
		spaceAABB.min_pos.x = min(spaceAABB.min_pos.x, aabb.min_pos.x);
		spaceAABB.min_pos.y = min(spaceAABB.min_pos.y, aabb.min_pos.y);
		spaceAABB.min_pos.z = min(spaceAABB.min_pos.z, aabb.min_pos.z);

		spaceAABB.max_pos.x = max(spaceAABB.max_pos.x, aabb.max_pos.x);
		spaceAABB.max_pos.y = max(spaceAABB.max_pos.y, aabb.max_pos.y);
		spaceAABB.max_pos.z = max(spaceAABB.max_pos.z, aabb.max_pos.z);
		////////////////////


		i++;
	}

	//hst_node.resize(vec_geoms.size()*2.5);

	vector<int> vec_sequence;	//geom_idx, used to rebuild a scene->geoms vector whose sequence = tree travse

	root_idx = build(vec_geoms, vec_sequence, spaceAABB, -1, 0);

	//rebuild scene->geoms according to vec_sequence;
	for (auto geom_idx : vec_sequence)
	{
		geoms_final.push_back(geoms_using.at(geom_idx));
	}
	geoms_using.clear();


}



void KDTree::buildLeaf(Node & cur
	,const vector<KDNodeConstructWrapper>& construct_objs
	, vector<int> & sequence, const AABB& box, int parent_idx, int depth)
{

	auto t = construct_objs.begin();
	cur.aabb = box;//t->aabb;

	//cur.geom_index = t->geom_idx;
	cur.geom_index = sequence.size();

	cur.parent_idx = parent_idx;
	//cur.left_idx = -1;
	cur.num_geoms = construct_objs.size();

	cur.right_idx = -1;

	for (auto c : construct_objs)
	{
		sequence.push_back(c.geom_idx);
	}

	//return cur_idx;
}







//return this node idx
int KDTree::build(vector<KDNodeConstructWrapper> & construct_objs
	, vector<int> & sequence, const AABB& box, int parent_idx, int depth)
{
	if (construct_objs.empty())
	{
		std::cerr << "ERROR: empty kdtree node!\n";
		return -1;
	}


	//if (last_idx >= hst_node.size())
	//{
	//	hst_node.push_back(Node());
	//}
	hst_node.push_back(Node());
	Node & cur = hst_node.at(last_idx); // !!! this is not safe when hst_node assigns new value
	int cur_idx = last_idx;
	last_idx++;


	if (construct_objs.size() <= MAX_LEAF_GEOM_NUM)
	{
		//leaf node
		//no more split
		buildLeaf(cur, construct_objs, sequence, box, parent_idx, depth);

		return cur_idx;
	}



	//internal node

	KdConstructCompareFun f;
	switch (depth % 3)
	{
	case 0:
		f = my_kd_construct_compare_x;
		cur.split.axis = AXIS_X;
		break;
	case 1:
		f = my_kd_construct_compare_y;
		cur.split.axis = AXIS_Y;
		break;
	case 2:
		f = my_kd_construct_compare_z;
		cur.split.axis = AXIS_Z;
		break;
	}


	//std::nth_element(construct_objs.begin(),construct_objs.begin()+(construct_objs.size()/2),construct_objs.end(),*f);

	sort(construct_objs.begin(), construct_objs.end(), *f);

	vector<KDNodeConstructWrapper>::iterator t = construct_objs.begin() + (construct_objs.size() / 2);

	cur.split.pos = t->mid[cur.split.axis];
	cur.aabb = box;
	cur.geom_index = -1;
	cur.parent_idx = parent_idx;
	pair<AABB, AABB> aabb_pair = cutAABB(box, cur.split);

	vector<KDNodeConstructWrapper> left_objs;
	vector<KDNodeConstructWrapper> right_objs;

	left_objs.assign(construct_objs.begin(), t);
	right_objs.assign(t, construct_objs.end());

	int tmp_left_size = left_objs.size();



	//overlap object should be added to both branch

	int num_overlap = 0;

	//add right to left
	for (auto o : right_objs)
	{
		if (o.aabb.min_pos[cur.split.axis] < cur.split.pos)
		{
			left_objs.push_back(o);
			num_overlap++;
		}
	}

	//add left to right
	for (int i = 0; i < tmp_left_size; i++)	//naive parse method....
	{
		KDNodeConstructWrapper & o = left_objs.at(i);
		if (o.aabb.max_pos[cur.split.axis] > cur.split.pos)
		{
			right_objs.push_back(o);
			num_overlap++;
		}
	}


	if ( ((double)num_overlap) / ((double)construct_objs.size()) >= MAX_OVERLAP_RATIO)
	{
		//don't split
		//build leaf
		buildLeaf(cur, construct_objs, sequence, box, parent_idx, depth);

		return cur_idx;
	}


	//cur.left_idx = build(left_objs, aabb_pair.first, cur_idx, depth + 1);
	build(left_objs, sequence, aabb_pair.first, cur_idx, depth + 1);
	
	hst_node.at(cur_idx).right_idx = build(right_objs, sequence, aabb_pair.second, cur_idx, depth + 1);
	


	return cur_idx;
}