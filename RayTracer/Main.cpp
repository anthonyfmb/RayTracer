#include <iostream>
#include <vector>
#include <fstream>
#include <string>
#include <sstream>

#define _CRT_SECURE_NO_WARNINGS

#include "glm/glm.hpp"
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STBI_MSC_SECURE_CRT
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

const float DimX = 800;
const float DimY = 800;

unsigned char image[800 * 800 * 3];

glm::vec3 up(0.0, 1.0, 0.0);
glm::vec3 eye(0.0, 0.0, 1.0);
glm::vec3 lookAt(0.0, 0.0, -1.0);

glm::vec3 l = glm::normalize(lookAt - eye);
glm::vec3 v = glm::normalize(glm::cross(l, up));
glm::vec3 u = glm::cross(v, l);

float fovy = atan((DimY / 2));

float a = DimX / DimY;
float d = 1 / tan(fovy / 2);

glm::vec3 ll = eye + d * l - a * v - u;

glm::vec3 background(0.0, 0.0, 0.0);
int maxDepth = 0;

class Ray 
{
public:
	glm::vec3 origin;
	glm::vec3 direction;

	Ray(glm::vec3 origin, glm::vec3 direction)
	{
		this->origin = origin;
		this->direction = direction;
	}
};

class Object {
public:
	glm::vec3 diff;
	glm::vec3 spec;
	float shininess;

	virtual glm::vec3 GetDiff()
	{
		return glm::vec3();
	}
};


class Sphere
{
public:
	glm::vec3 center;
	float radius;
	glm::vec3 diff;
	glm::vec3 spec;
	float shininess;

	Sphere(glm::vec3 center, float radius, glm::vec3 diff, glm::vec3 spec, float shininess)
	{
		this->center = center;
		this->radius = radius;
		this->shininess = shininess;
		this->spec = spec;
		this->diff = diff;
	}

	Sphere() {}

	glm::vec3 GetDiff()
	{
		return diff;
	}

};

class Triangle
{
public: 
	glm::vec3 a;
	glm::vec3 b;
	glm::vec3 c;
	glm::vec3 diff;
	glm::vec3 spec;
	float shininess = 0.0f;

	Triangle(glm::vec3 a, glm::vec3 b, glm::vec3 c, glm::vec3 diff, glm::vec3 spec, float shininess)
	{
		this->a = a;
		this->b = b;
		this->c = c;
		this->diff = diff;
		this->spec = spec;
		this->shininess = shininess;
	}

	Triangle()
	{
		a, b, c = glm::vec3(0.0, 0.0, 0.0);
	}


	glm::vec3 GetDiff()
	{
		return diff;
	}

};

class Quad
{
public:
	glm::vec3 a;
	glm::vec3 b;
	glm::vec3 c;
	glm::vec3 diff;
	glm::vec3 spec;
	float shininess;

	Quad(glm::vec3 a, glm::vec3 b, glm::vec3 c, glm::vec3 diff, glm::vec3 spec, float shininess)
	{
		this->a = a;
		this->b = b;
		this->c = c;
		this->diff = diff;
		this->spec = spec;
		this->shininess = shininess;
	}

	Quad() {}


	glm::vec3 GetDiff()
	{
		return diff;
	}
};

class Intersection
{
public:
	glm::vec3 pos;
	glm::vec3 diff;
	glm::vec3 spec;
	glm::vec3 normal;
	float shininess;

	Intersection(glm::vec3 pos, glm::vec3 diff, glm::vec3 spec, glm::vec3 normal, float shininess)
	{
		this->pos = pos;
		this->diff = diff;
		this->spec = spec;
		this->normal = normal;
		this->shininess = shininess;
	}

	Intersection()
	{

	}
};


class Light
{
public:
	glm::vec3 pos;
	glm::vec3 diff;
	glm::vec3 spec;

	Light(glm::vec3 pos, glm::vec3 diff, glm::vec3 spec)
	{
		this->pos = pos;
		this->diff = diff;
		this->spec = spec;
	}
	Light() {}
};


std::vector<Sphere> spheres;
std::vector<Triangle> triangles;
std::vector<Light> lights;
std::vector<Quad> quads;

/*
The way I'm parsing the file is awful, sorry if you're reading this
*/
void Parse(std::istream& istr)
{
	std::string l;
	std::string val;
	while (std::getline(istr, l))
	{

		if (l.compare("LIGHT") == 0)
		{
			Light light;
			std::getline(istr, l);
			std::istringstream iss(l);
			glm::vec3 p;
			iss >> val;
			iss >> val;
			light.pos.x = std::stof(val);
			iss >> val;
			light.pos.y = std::stof(val);
			iss >> val;
			light.pos.z = std::stof(val);

			std::getline(istr, l);
			iss.clear();
			iss.str(l);
			iss >> val;
			iss >> val;
			light.diff.x = std::stof(val);
			iss >> val;
			light.diff.y = std::stof(val);
			iss >> val;
			light.diff.z = std::stof(val);

			std::getline(istr, l);
			iss.clear();
			iss.str(l);
			iss >> val;
			iss >> val;
			light.spec.x = std::stof(val);
			iss >> val;
			light.spec.y = std::stof(val);
			iss >> val;
			light.spec.z = std::stof(val);
			lights.push_back(light);
		}
		else if (l.compare("SPHERE") == 0)
		{
			Sphere s;
			std::getline(istr, l);
			std::istringstream iss(l);
			iss >> val;
			iss >> val;
			s.center.x = std::stof(val);
			iss >> val;
			s.center.y = std::stof(val);
			iss >> val;
			s.center.z = std::stof(val);

			std::getline(istr, l);
			iss.clear();
			iss.str(l);
			iss >> val;
			iss >> val;
			s.radius = std::stof(val);

			std::getline(istr, l);
			iss.clear();
			iss.str(l);
			iss >> val;
			iss >> val;
			s.diff.x = std::stof(val);
			iss >> val;
			s.diff.y = std::stof(val);
			iss >> val;
			s.diff.z = std::stof(val);

			std::getline(istr, l);
			iss.clear();
			iss.str(l);
			iss >> val;
			iss >> val;
			s.spec.x = std::stof(val);
			iss >> val;
			s.spec.y = std::stof(val);
			iss >> val;
			s.spec.z = std::stof(val);

			std::getline(istr, l);
			iss.clear();
			iss.str(l);
			iss >> val;
			iss >> val;
			s.shininess = std::stof(val);
			spheres.push_back(s);
		}
		else if (l.compare("QUAD") == 0)
		{
			Quad q;
			// Point A
			std::getline(istr, l);
			std::istringstream iss(l);
			iss >> val;
			iss >> val;
			q.a.x = std::stof(val);
			iss >> val;
			q.a.y = std::stof(val);
			iss >> val;
			q.a.z = std::stof(val);

			// Point B
			std::getline(istr, l);
			iss.clear();
			iss.str(l);
			iss >> val;
			iss >> val;
			q.b.x = std::stof(val);
			iss >> val;
			q.b.y = std::stof(val);
			iss >> val;
			q.b.z = std::stof(val);

			// Point C
			std::getline(istr, l);
			iss.clear();
			iss.str(l);
			iss >> val;
			iss >> val;
			q.c.x = std::stof(val);
			iss >> val;
			q.c.y = std::stof(val);
			iss >> val;
			q.c.z = std::stof(val);

			// Diffuse
			std::getline(istr, l);
			iss.clear();
			iss.str(l);
			iss >> val;
			iss >> val;
			q.diff.x = std::stof(val);
			iss >> val;
			q.diff.y = std::stof(val);
			iss >> val;
			q.diff.z = std::stof(val);

			// Specular
			std::getline(istr, l);
			iss.clear();
			iss.str(l);
			iss >> val;
			iss >> val;
			q.spec.x = std::stof(val);
			iss >> val;
			q.spec.y = std::stof(val);
			iss >> val;
			q.spec.z = std::stof(val);

			// Shininess
			std::getline(istr, l);
			iss.clear();
			iss.str(l);
			iss >> val;
			iss >> val;
			q.shininess = std::stof(val);
			quads.push_back(q);
		}
		else if (l.compare("TRIANGLE") == 0)
		{
			Triangle q;
			// Point A
			std::getline(istr, l);
			std::istringstream iss(l);
			iss >> val;
			iss >> val;
			q.a.x = std::stof(val);
			iss >> val;
			q.a.y = std::stof(val);
			iss >> val;
			q.a.z = std::stof(val);

			// Point B
			std::getline(istr, l);
			iss.clear();
			iss.str(l);
			iss >> val;
			iss >> val;
			q.b.x = std::stof(val);
			iss >> val;
			q.b.y = std::stof(val);
			iss >> val;
			q.b.z = std::stof(val);

			// Point C
			std::getline(istr, l);
			iss.clear();
			iss.str(l);
			iss >> val;
			iss >> val;
			q.c.x = std::stof(val);
			iss >> val;
			q.c.y = std::stof(val);
			iss >> val;
			q.c.z = std::stof(val);

			// Diffuse
			std::getline(istr, l);
			iss.clear();
			iss.str(l);
			iss >> val;
			iss >> val;
			q.diff.x = std::stof(val);
			iss >> val;
			q.diff.y = std::stof(val);
			iss >> val;
			q.diff.z = std::stof(val);

			// Specular
			std::getline(istr, l);
			iss.clear();
			iss.str(l);
			iss >> val;
			iss >> val;
			q.spec.x = std::stof(val);
			iss >> val;
			q.spec.y = std::stof(val);
			iss >> val;
			q.spec.z = std::stof(val);

			// Shininess
			std::getline(istr, l);
			iss.clear();
			iss.str(l);
			iss >> val;
			iss >> val;
			q.shininess = std::stof(val);
			triangles.push_back(q);
		}
		else if (l.compare("BACKGROUND") == 0)
		{
			std::getline(istr, l);
			std::istringstream iss(l);
			iss >> val;
			iss >> val;
			background.r = std::stof(val);
			iss >> val;
			background.g = std::stof(val);
			iss >> val;
			background.b = std::stof(val);
		}
		else if (l.compare("MAXDEPTH") == 0)
		{
			std::getline(istr, l);
			maxDepth = std::stoi(l);
		}
	}
}

void ParseFile(std::string filename) {
	std::ifstream file(filename);

	return Parse(file);
}


/*
Source: 
The wiki article linked on the slides
https://en.wikipedia.org/wiki/M%C3%B6ller%E2%80%93Trumbore_intersection_algorithm
*/
bool TriangleIntersection(
	glm::vec3 ray_origin,
	glm::vec3 ray_vector,
	const Triangle& triangle,
	glm::vec3& out_intersection_point)
{
	constexpr float epsilon = std::numeric_limits<float>::epsilon();

	glm::vec3 edge1 = triangle.b - triangle.a;
	glm::vec3 edge2 = triangle.c - triangle.a;
	glm::vec3 ray_cross_e2 = glm::cross(ray_vector, edge2);
	float det = glm::dot(edge1, ray_cross_e2);

	if (det > -epsilon && det < epsilon)
		return false;    // This ray is parallel to this triangle.

	float inv_det = 1.0 / det;
	glm::vec3 s = ray_origin - triangle.a;
	float u = inv_det * glm::dot(s, ray_cross_e2);

	if (u < 0 || u > 1)
		return false;

	glm::vec3 s_cross_e1 = glm::cross(s, edge1);
	float v = inv_det * glm::dot(ray_vector, s_cross_e1);

	if (v < 0 || u + v > 1)
		return false;

	// At this stage we can compute t to find out where the intersection point is on the line.
	float t = inv_det * glm::dot(edge2, s_cross_e1);

	if (t > epsilon) // ray intersection
	{
		out_intersection_point = ray_origin + ray_vector * t;
		return true;
	}
	else // This means that there is a line intersection but not a ray intersection.
		return false;
}

bool SphereIntersection(Ray ray, Sphere sphere, glm::vec3& out_intersection_point)
{
	glm::vec3 v = ray.origin - sphere.center;

	glm::vec3 dir = ray.direction;
	glm::vec3 origin = ray.origin;
	glm::vec3 center = sphere.center;

	float a = 1;
	float b = 2 * (dir.x * (origin.x - center.x) +
		dir.y * (origin.y - center.y) + dir.z * (origin.z - center.z));
	float c = std::pow((origin.x - center.x), 2) + 
		std::pow((origin.y - center.y), 2) + std::pow((origin.z - center.z), 2)
		- std::pow(sphere.radius, 2);
	
	float discriminant = b * b - 4 * a * c;

	if (discriminant >= 0) {
		float t1 = (-b - sqrt(discriminant)) / (2.0 * a);

		if (t1 > 0) {
			out_intersection_point = ray.origin + t1 * ray.direction;
			return true;
		}
	}
	return false;
}

/*
Just the triangle intersection with small modification
*/
bool QuadIntersection(Ray ray, Quad quad, glm::vec3& out_intersection_point)
{
	constexpr float epsilon = std::numeric_limits<float>::epsilon();

	glm::vec3 edge1 = quad.b - quad.a;
	glm::vec3 edge2 = quad.c - quad.a;
	glm::vec3 ray_cross_e2 = glm::cross(ray.direction, edge2);
	float det = glm::dot(edge1, ray_cross_e2);

	if (det > -epsilon && det < epsilon)
		return false;    // This ray is parallel to this triangle.

	float inv_det = 1.0 / det;
	glm::vec3 s = ray.origin - quad.a;
	float u = inv_det * glm::dot(s, ray_cross_e2);

	if (u < 0 || u > 1)
		return false;

	glm::vec3 s_cross_e1 = glm::cross(s, edge1);
	float v = inv_det * glm::dot(ray.direction, s_cross_e1);

	if (v < 0 || v > 1)
		return false;

	// At this stage we can compute t to find out where the intersection point is on the line.
	float t = inv_det * glm::dot(edge2, s_cross_e1);

	if (t > epsilon) // ray intersection
	{
		out_intersection_point = ray.origin + ray.direction * t;
		return true;
	}
	else // This means that there is a line intersection but not a ray intersection.
		return false;
}

Ray CalculateRay(float i, float j)
{
	glm::vec3 p = ll + 2 * a * v * i / DimX + 2.f * u * j / DimY;
	glm::vec3 d = glm::normalize(p - eye);
	Ray r = Ray(eye, d);
	return r;
}

/*
Calculates the first intersection with an object (not a light)
If there's no intersection, it returns glm::vec3(INFINITY, INFINITY, INFINITY)
*/
glm::vec3 FirstIntersection(glm::vec3 p, Ray ray, Intersection* intersect)
{
	glm::vec3 closestIntersection = glm::vec3(INFINITY, INFINITY, INFINITY);
	glm::vec3 intersection;
	for  (Triangle& t : triangles)
	{
		if (TriangleIntersection(ray.origin, ray.direction, t, intersection)) 
		{
			if (glm::length(intersection - p) < glm::length(closestIntersection - p))
			{
				closestIntersection = intersection;
				intersect->diff = t.diff;
				intersect->spec = t.spec;
				intersect->pos = intersection;
				intersect->shininess = t.shininess;

				glm::vec3 normal = glm::normalize(glm::cross(t.b - t.a, t.c - t.a));
				if (glm::dot(ray.direction, normal) > 0)
				{
					normal = -1.f * normal;
				}
				intersect->normal = normal;
			}
		}
	}

	for (Sphere& s : spheres)
	{
		if (SphereIntersection(ray, s, intersection))
		{
			if (glm::length(intersection - p) < glm::length(closestIntersection - p))
			{
				closestIntersection = intersection;
				intersect->diff = s.diff;
				intersect->spec = s.spec;
				intersect->pos = intersection;
				intersect->shininess = s.shininess;
				glm::vec3 normal = glm::normalize(intersection - s.center);
				intersect->normal = normal;
			}
		}
	}

	for (Quad& q : quads)
	{
		if (QuadIntersection(ray, q, intersection))
		{
			if (glm::length(intersection - p) < glm::length(closestIntersection - p))
			{
				closestIntersection = intersection;
				intersect->diff = q.diff;
				intersect->spec = q.spec;
				intersect->pos = intersection;
				intersect->shininess = q.shininess;
				glm::vec3 normal = glm::normalize(glm::cross(q.b - q.a, q.c - q.a));
				if (glm::dot(ray.direction, normal) > 0)
				{
					normal = -1.f * normal;
				}
				intersect->normal = normal;
			}
		}
	}

	return closestIntersection;
}

/*
Calculates which lights are hitting the point p
*/
std::vector<Light> ShadowRays(glm::vec3 p) 
{
	std::vector<Light> contributedLights;
	// Iterate through each light source and see if they hit this point
	for (const Light& l : lights)
	{
		glm::vec3 dir = glm::normalize(l.pos - p);
		Ray r = Ray(p + 0.001f * dir, dir);

		Intersection intersect;
		// Check if the ray hits an object before the light
		glm::vec3 intersection = FirstIntersection(p, r, &intersect);


		// If the intersection with the closest object happens after 
		// the intersection with the light, then we hit the light first 
		if (glm::length(intersection - p) > glm::length(l.pos - p))
		{
			contributedLights.push_back(l);
		}
	}

	return contributedLights;
}


Ray ReflectedRay(Ray r, glm::vec3 p, glm::vec3 normal)
{	
	Ray ray = Ray(p, glm::reflect(r.direction, normal));

	return ray;
}

/*
Apply Phong illumination model
*/
glm::vec3 Phong(glm::vec3 p, Light l, Intersection intersect, glm::vec3 view)
{
	glm::vec3 objectDiff = intersect.diff;
	glm::vec3 objectSpec = intersect.spec;
	glm::vec3 normal = intersect.normal;
	float objectShine = intersect.shininess;

	float diffuse = glm::dot(normal, glm::normalize(l.pos - p));
	if (diffuse < 0)
		diffuse = 0;
	glm::vec3 diffuseComponent = diffuse * l.diff * objectDiff;

	glm::vec3 I = glm::normalize(l.pos - p);
	glm::vec3 r = I - 2.f * (glm::dot(I, normal)) * normal;
	float specular = glm::dot(r, view);
	if (specular < 0)
		specular = 0;
	else
		specular = glm::pow(specular, objectShine);
	glm::vec3 specularComponent = specular * l.spec * objectSpec;

	glm::vec3 phong = specularComponent + diffuseComponent;
	
	return phong;
}

/*
Apply Phong illumination model for each contributing light
Recursively calculate reflections
*/
glm::vec3 TraceRay(Ray ray, int depth) {
	Intersection intersect;
	glm::vec3 p = FirstIntersection(eye, ray, &intersect); //get the first one

	if (p == glm::vec3(INFINITY, INFINITY, INFINITY))
	{
		return background; 
	}
	
	std::vector<Light> contributedLights = ShadowRays(p);

	glm::vec3 color = glm::vec3(0,0,0);

	for (int i = 0; i < contributedLights.size(); i++)
	{
		color += Phong(p, contributedLights[i], intersect, ray.direction);
	}

	if (depth <= 0 || intersect.spec == glm::vec3(0.0, 0.0, 0.0) || contributedLights.size() == 0)
	{
		return color;
	}

	Ray reflected = ReflectedRay(ray, p, intersect.normal);
	color += TraceRay(reflected, depth - 1) * 0.5f;

	return color;
}

void RenderImage(const char* filename)
{
	/*
	Example Scene without file reading:

	Sphere s = Sphere(glm::vec3(0.0, 0, -12.f), 2.f, glm::vec3(1.0, 1.0, 1.0), glm::vec3(1.0, 1.0, 1.0), 10.0);
	Light l = Light(glm::vec3(0.0, 5.0, -8.f), glm::vec3(1.0, 1.0, 1.0), glm::vec3(1.0, 1.0, 1.0));
	Triangle t = Triangle(glm::vec3(-1, 0, -18.f), glm::vec3(-1, 0, -14.f), glm::vec3(-1, 1, -14.f), glm::vec3(0.0, 0.0, 0.0), glm::vec3(0,0,0), 0.0f);
	Quad q = Quad(glm::vec3(-20, -10, -18.f), glm::vec3(-20, 10, -18.f), glm::vec3(20, -10, -18.f), glm::vec3(0.0, 1.0, 1.0), glm::vec3(0, 0, 0), 10.0f);
	Quad qleft = Quad(glm::vec3(-10, -10, -5.f), glm::vec3(-10, 10, -5.f), glm::vec3(-10, -10, -18.f), glm::vec3(1.0, 0.0, 0.0), glm::vec3(0, 0, 0), 0.0f);
	Quad qright = Quad(glm::vec3(10, -10, -5.f), glm::vec3(10, 10, -5.f), glm::vec3(10, -10, -18.f), glm::vec3(0.0, 1.0, 0.0), glm::vec3(0.0, 0.0, 0.0), 10.0f);
	Quad qtop = Quad(glm::vec3(-10, 10, -5.f), glm::vec3(-10, 10, -18.f), glm::vec3(10, 10, -5.f), glm::vec3(0.0, 0.0, 1.0), glm::vec3(0, 0, 0), 0.0f);
	Quad qbottom = Quad(glm::vec3(-10, -10, -5.f), glm::vec3(-10, -10, -18.f), glm::vec3(10, -10, -5.f), glm::vec3(1.0, 1.0, 0.0), glm::vec3(0, 0, 0), 0.0f);
	
	spheres.push_back(s);
	lights.push_back(l);
	triangles.push_back(t);
	quads.push_back(q);
	quads.push_back(qleft);
	quads.push_back(qright);
	quads.push_back(qbottom);
	quads.push_back(qtop);
	*/

	int index = 0;
	for (int i = 0; i < DimX; i++) {
		for (int j = 0; j < DimY; j++) {
			Ray ray = CalculateRay(j, DimX - i - 1);

			glm::vec3 color = TraceRay(ray, maxDepth);
			color = glm::vec3(255, 255, 255) * color;

			if (color.r > 255) color.r = 255;
			if (color.g > 255) color.g = 255;
			if (color.b > 255) color.b = 255;

			image[index++] = static_cast<unsigned char>(color.r);
			image[index++] = static_cast<unsigned char>(color.g);
			image[index++] = static_cast<unsigned char>(color.b);
		}
	}

	stbi_write_png(filename, 800, 800, 3, image, 800 * 3);
}

int main(int argc, char *argv[])
{
	ParseFile("test.txt");
	RenderImage("image.png");
}

