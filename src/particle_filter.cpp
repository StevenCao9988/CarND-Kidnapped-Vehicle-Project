/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 */

#include "particle_filter.h"

#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "helper_functions.h"

using std::string;
using std::vector;
using namespace std;

static default_random_engine gen;  // ����һ����������Թ���������ʹ��
#define EPS 0.00001    // ����һ���ǳ�С����

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  /**
   * TODO: Set the number of particles. Initialize all particles to 
   *   first position (based on estimates of x, y, theta and their uncertainties
   *   from GPS) and all weights to 1. 
   * TODO: Add random Gaussian noise to each particle.
   * NOTE: Consult particle_filter.h for more information about this method 
   *   (and others in this file).
   */
  num_particles = 100;  // TODO: Set the number of particles

  // Ϊ������������������̬�ֲ�
  normal_distribution<double> noise_x(0, std[0]);
  normal_distribution<double> noise_y(0, std[1]);
  normal_distribution<double> noise_theta(0, std[2]);
 
  // ��ʼ��������
  for (int i = 0; i < num_particles; i++)
  {
	  Particle p;
	  p.id = i;
	  p.x = x;
	  p.y = y;
	  p.theta = theta;
	  p.weight = 1.0;

	  // �������
	  p.x = p.x + noise_x(gen);
	  p.y = p.y + noise_y(gen);
	  p.theta = p.theta + noise_theta(gen);

	  // ��ӵ����Ӽ��ϵ����һλ
	  particles.push_back(p);
  }//  

  is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], 
                                double velocity, double yaw_rate) {
  /**
   * TODO: Add measurements to each particle and add random Gaussian noise.
   * NOTE: When adding noise you may find std::normal_distribution 
   *   and std::default_random_engine useful.
   *  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
   *  http://www.cplusplus.com/reference/random/default_random_engine/
   */
	normal_distribution<double> noise_x(0, std_pos[0]);
	normal_distribution<double> noise_y(0, std_pos[1]);
	normal_distribution<double> noise_theta(0, std_pos[2]);

	for (int i = 0; i < num_particles; i++)
	{
		if (fabs(yaw_rate) > EPS)// ת���˶�
		{
			// �뾶 = �ٶ� / ���ٶ�
			double R = velocity / yaw_rate;
			// Ԥ�⳵���˹��ĽǶ�
			double new_theta = particles[i].theta + yaw_rate * delta_t;

			particles[i].x += R * (sin(new_theta) - sin(particles[i].theta));
			particles[i].y += R * (cos(particles[i].theta) - cos(new_theta));
			particles[i].theta = new_theta;
		}
		else// ֱ���˶�
		{
			particles[i].x += velocity * delta_t * cos(particles[i].theta);
			particles[i].y += velocity * delta_t * sin(particles[i].theta);
		}// if (yaw_rate > 0.00001) ����

		// �������
		particles[i].x = particles[i].x + noise_x(gen);
		particles[i].y = particles[i].y + noise_y(gen);
		particles[i].theta = particles[i].theta + noise_theta(gen);
	}

}

void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted,
                                     vector<LandmarkObs>& observations) {
  /**
   * TODO: Find the predicted measurement that is closest to each 
   *   observed measurement and assign the observed measurement to this 
   *   particular landmark.
   * NOTE: this method will NOT be called by the grading code. But you will 
   *   probably find it useful to implement this method and use it as a helper 
   *   during the updateWeights phase.
   */

	// ����ÿһ���۲�����
	for (int i = 0; i < observations.size(); i++)
	{
		// ��ȡ���Ƶ�·��λ�ã�ͨ���������ݣ�
		LandmarkObs obs = observations[i];

		// ����ʵ��·���������·���������
		double min_dist = numeric_limits<double>::max();

		// ���������������ݣ����ײ������ĸ�·��
		int id_in_map = -100;

		// ����ÿһ��·��
		for (int j = 0; j < predicted.size(); j++)
		{
			double dis = dist(observations[i].x, observations[i].y, predicted[j].x, predicted[j].y);
			// ��ʵ��·���У�Ѱ���뷴��·�������С�� ���㷨���ԸĽ���
			if (dis < min_dist)
			{
				min_dist = dis;
				id_in_map = predicted[j].id;
			}
		}//����������ÿһ��·��

		// �����������ݣ��⵽�����ĸ�·��
		observations[i].id = id_in_map;
	}// ����������ÿһ���۲�����

}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
                                   const vector<LandmarkObs> &observations, 
                                   const Map &map_landmarks) {
  /**
   * TODO: Update the weights of each particle using a mult-variate Gaussian 
   *   distribution. You can read more about this distribution here: 
   *   https://en.wikipedia.org/wiki/Multivariate_normal_distribution
   * NOTE: The observations are given in the VEHICLE'S coordinate system. 
   *   Your particles are located according to the MAP'S coordinate system. 
   *   You will need to transform between the two systems. Keep in mind that
   *   this transformation requires both rotation AND translation (but no scaling).
   *   The following is a good resource for the theory:
   *   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
   *   and the following is a good resource for the actual equation to implement
   *   (look at equation 3.33) http://planning.cs.uiuc.edu/node99.html
   */
	// x����y�᷽��Ĳ�ȷ����
	double stdX= std_landmark[0];
	double stdY = std_landmark[1];

	// ����ÿһ������ particles 
	for (int i = 0; i < num_particles; i++)
	{
		// �õ�������ӵ�����
		double p_x = particles[i].x;
		double p_y = particles[i].y;
		double p_theta = particles[i].theta;


		//  ����ÿһ��·��
		vector<LandmarkObs> inRangeLandmarks;    // ��������ӷ�Χ�ڵ�·�꼯��
		for (int j = 0; j < map_landmarks.landmark_list.size(); j++)
		{
			// �õ����·�������
			int lm_id = map_landmarks.landmark_list[j].id_i;
			float lm_x = map_landmarks.landmark_list[j].x_f;
			float lm_y = map_landmarks.landmark_list[j].y_f;
			double dX = p_x - lm_x;
			double dY = p_y - lm_y;
			double sensor_range_2 = sensor_range * sensor_range;
			
			// ֻ�������Ӹ�����·�����Ч����ʱ�Դ�����������ΧΪ��׼��
			// Ϊ�򻯼��㣬�ſ��׼��ֻ���������С����
			if (dX * dX + dY * dY <= sensor_range_2)
			{
				inRangeLandmarks.push_back(LandmarkObs{lm_id, lm_x, lm_y});
			}
		}//������  ����ÿһ��·��

		// ����ÿһ���۲�����  ����ת��(��������۲����ݶ�Ӧ��·��λ��)
		vector<LandmarkObs> mappedObservations;    // ��������ӷ�Χ�ڵ�·�꼯��
		for (int j = 0; j < observations.size(); j++)
		{
			int ob_id = observations[j].id;  // �������ĸ�·�꣬����Ҫ��������dataAssociation����
			double ob_x = p_x + observations[j].x * cos(p_theta) - observations[j].y * sin(p_theta);
			double ob_y = p_y + observations[j].x * sin(p_theta) + observations[j].y * cos(p_theta);
			mappedObservations.push_back(LandmarkObs{ob_id, ob_x, ob_y});
		}//����������ÿһ���۲�����  ����ת��

		// �Ʋ�۲����ݣ������ǹ۲���ĸ�·�꣬�洢��mappedObservations.id��
		dataAssociation(inRangeLandmarks, mappedObservations);

		// ��ʼ����������ӵı��أ� ���ȳ�ʼ��Ϊ1
		particles[i].weight = 1.0;
		
		// ����ÿһ���������ݣ� ����������ӵı���
		for (int j = 0; j < mappedObservations.size(); j++)
		{
			double observationX = mappedObservations[j].x;
			double observationY = mappedObservations[j].y;
			double landmarkX = 0;
			double landmarkY = 0;

			// �ҳ�����������ݶ�Ӧ��·��
			for (int k = 0; k < inRangeLandmarks.size(); k++)
			{
				if (inRangeLandmarks[k].id == mappedObservations[j].id)
				{
					landmarkX = inRangeLandmarks[k].x;
					landmarkY = inRangeLandmarks[k].y;
					break;     // ����
				}	
			}// �������ҳ�����������ݶ�Ӧ��·��

			// ��ÿ���������ݵı������
			double dX = observationX - landmarkX;
			double dY = observationY - landmarkY;

			// ���±���
			double weight = (1 / (2 * M_PI * stdX * stdY)) * exp(-(dX * dX / (2 * stdX * stdX) + (dY * dY / (2 * stdY * stdY)) ));

			//if weight equal to zero. then multiply to the EPS. 
			if (weight == 0) 
			{
				particles[i].weight = particles[i].weight * EPS;
			}
			//if weight doesn't equal to zero, then weight should be multiply by i times. because it is multivariate define.
			else
			{
				particles[i].weight = particles[i].weight * weight;
			}
			
		}// ����������ÿһ���������ݣ�����������ӵı���
	}//����������ÿһ������ particles

}

void ParticleFilter::resample() {
  /**
   * TODO: Resample particles with replacement with probability proportional 
   *   to their weight. 
   * NOTE: You may find std::discrete_distribution helpful here.
   *   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
   */

	// ����100�����������ѡȡ100�Σ��ø��ʸ����������ӱ����
	// ���������������������ӷ��ڣ� �������̲�ͬ��������Ǵ���ͬ������
	vector<double> weights;
	double max_weight = numeric_limits<double>::min();

	for (int i = 0; i < num_particles; i++)
	{
		weights.push_back(particles[i].weight);
		if (particles[i].weight > max_weight)
		{
			max_weight = particles[i].weight;
		}
	}

	// ����ƽ���������
	uniform_real_distribution<float> dist_float(0.0, max_weight);
	uniform_int_distribution<int> dist_int(0, num_particles - 1);

	// �����ʼλ��
	int index = dist_int(gen);

	// ��ʼ�ӷ���
	float beta = 0.0;
	
	vector<Particle> ResampleParticle;
	for (int i = 0; i < num_particles; i++)
	{
		// 1���ӳ�ȥ
		beta = beta + dist_float(gen) * 2.0;

		// 2������һ��������һ��
		while (beta > weights[index])            
		{
			beta = beta - weights[index];
			index = (index + 1) % num_particles;   // ǰ��һ��
		}
		ResampleParticle.push_back(particles[index]);
	}
	// �ӷ��ڽ���
	particles = ResampleParticle; 

}

void ParticleFilter::SetAssociations(Particle& particle, 
                                     const vector<int>& associations, 
                                     const vector<double>& sense_x, 
                                     const vector<double>& sense_y) {
  // particle: the particle to which assign each listed association, 
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
	//Clear the previous associations
	particle.associations.clear();
	particle.sense_x.clear();
	particle.sense_y.clear();
	particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord) {
  vector<double> v;

  if (coord == "X")
  {
    v = best.sense_x;
  } else 
  {
    v = best.sense_y;
  }

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}