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

static default_random_engine gen;  // 定义一个随机数，以供其他函数使用
#define EPS 0.00001    // 定义一个非常小的数

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

  // 为传感器噪声，创建正态分布
  normal_distribution<double> noise_x(0, std[0]);
  normal_distribution<double> noise_y(0, std[1]);
  normal_distribution<double> noise_theta(0, std[2]);
 
  // 初始化粒子们
  for (int i = 0; i < num_particles; i++)
  {
	  Particle p;
	  p.id = i;
	  p.x = x;
	  p.y = y;
	  p.theta = theta;
	  p.weight = 1.0;

	  // 添加噪音
	  p.x = p.x + noise_x(gen);
	  p.y = p.y + noise_y(gen);
	  p.theta = p.theta + noise_theta(gen);

	  // 添加到粒子集合的最后一位
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
		if (fabs(yaw_rate) > EPS)// 转向运动
		{
			// 半径 = 速度 / 角速度
			double R = velocity / yaw_rate;
			// 预测车辆运功的角度
			double new_theta = particles[i].theta + yaw_rate * delta_t;

			particles[i].x += R * (sin(new_theta) - sin(particles[i].theta));
			particles[i].y += R * (cos(particles[i].theta) - cos(new_theta));
			particles[i].theta = new_theta;
		}
		else// 直线运动
		{
			particles[i].x += velocity * delta_t * cos(particles[i].theta);
			particles[i].y += velocity * delta_t * sin(particles[i].theta);
		}// if (yaw_rate > 0.00001) 结束

		// 添加噪音
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

	// 分析每一个观测数据
	for (int i = 0; i < observations.size(); i++)
	{
		// 获取反推的路标位置（通过测量数据）
		LandmarkObs obs = observations[i];

		// 定义实际路标与测量的路标最近距离
		double min_dist = numeric_limits<double>::max();

		// 定义这条测量数据，到底测量的哪个路标
		int id_in_map = -100;

		// 分析每一个路标
		for (int j = 0; j < predicted.size(); j++)
		{
			double dis = dist(observations[i].x, observations[i].y, predicted[j].x, predicted[j].y);
			// 在实际路标中，寻找与反推路标距离最小的 （算法可以改进）
			if (dis < min_dist)
			{
				min_dist = dis;
				id_in_map = predicted[j].id;
			}
		}//结束：分析每一个路标

		// 这条测量数据，测到的是哪个路标
		observations[i].id = id_in_map;
	}// 结束：分析每一个观测数据

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
	// x轴与y轴方向的不确定性
	double stdX= std_landmark[0];
	double stdY = std_landmark[1];

	// 分析每一个粒子 particles 
	for (int i = 0; i < num_particles; i++)
	{
		// 得到这个粒子的数据
		double p_x = particles[i].x;
		double p_y = particles[i].y;
		double p_theta = particles[i].theta;


		//  分析每一个路标
		vector<LandmarkObs> inRangeLandmarks;    // 在这个粒子范围内的路标集合
		for (int j = 0; j < map_landmarks.landmark_list.size(); j++)
		{
			// 得到这个路标的数据
			int lm_id = map_landmarks.landmark_list[j].id_i;
			float lm_x = map_landmarks.landmark_list[j].x_f;
			float lm_y = map_landmarks.landmark_list[j].y_f;
			double dX = p_x - lm_x;
			double dY = p_y - lm_y;
			double sensor_range_2 = sensor_range * sensor_range;
			
			// 只有在粒子附近的路标才有效，暂时以传感器测量范围为标准，
			// 为简化计算，放宽标准，只考虑坐标大小即可
			if (dX * dX + dY * dY <= sensor_range_2)
			{
				inRangeLandmarks.push_back(LandmarkObs{lm_id, lm_x, lm_y});
			}
		}//结束：  分析每一个路标

		// 分析每一个观测数据  坐标转换(反推这个观测数据对应的路标位置)
		vector<LandmarkObs> mappedObservations;    // 在这个粒子范围内的路标集合
		for (int j = 0; j < observations.size(); j++)
		{
			int ob_id = observations[j].id;  // 到底用哪个路标，还需要经过函数dataAssociation计算
			double ob_x = p_x + observations[j].x * cos(p_theta) - observations[j].y * sin(p_theta);
			double ob_y = p_y + observations[j].x * sin(p_theta) + observations[j].y * cos(p_theta);
			mappedObservations.push_back(LandmarkObs{ob_id, ob_x, ob_y});
		}//结束：分析每一个观测数据  坐标转换

		// 推测观测数据，到底是观测的哪个路标，存储在mappedObservations.id里
		dataAssociation(inRangeLandmarks, mappedObservations);

		// 开始计算这个粒子的比重， 首先初始化为1
		particles[i].weight = 1.0;
		
		// 分析每一个测量数据， 计算这个粒子的比重
		for (int j = 0; j < mappedObservations.size(); j++)
		{
			double observationX = mappedObservations[j].x;
			double observationY = mappedObservations[j].y;
			double landmarkX = 0;
			double landmarkY = 0;

			// 找出这个测量数据对应的路标
			for (int k = 0; k < inRangeLandmarks.size(); k++)
			{
				if (inRangeLandmarks[k].id == mappedObservations[j].id)
				{
					landmarkX = inRangeLandmarks[k].x;
					landmarkY = inRangeLandmarks[k].y;
					break;     // 结束
				}	
			}// 结束：找出这个测量数据对应的路标

			// 将每个测量数据的比重相乘
			double dX = observationX - landmarkX;
			double dY = observationY - landmarkY;

			// 更新比重
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
			
		}// 结束：分析每一个测量数据，计算这个粒子的比重
	}//结束：分析每一个粒子 particles

}

void ParticleFilter::resample() {
  /**
   * TODO: Resample particles with replacement with probability proportional 
   *   to their weight. 
   * NOTE: You may find std::discrete_distribution helpful here.
   *   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
   */

	// 在这100个粒子里，重新选取100次，让概率更大的粒子脱颖而出
	// 可以想象成往靶子上随机扔飞镖， 而飞镖盘不同的区域就是代表不同的粒子
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

	// 建立平均的随机数
	uniform_real_distribution<float> dist_float(0.0, max_weight);
	uniform_int_distribution<int> dist_int(0, num_particles - 1);

	// 定义初始位置
	int index = dist_int(gen);

	// 开始扔飞镖
	float beta = 0.0;
	
	vector<Particle> ResampleParticle;
	for (int i = 0; i < num_particles; i++)
	{
		// 1、扔出去
		beta = beta + dist_float(gen) * 2.0;

		// 2、计算一下扔在哪一格
		while (beta > weights[index])            
		{
			beta = beta - weights[index];
			index = (index + 1) % num_particles;   // 前进一格
		}
		ResampleParticle.push_back(particles[index]);
	}
	// 扔飞镖结束
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