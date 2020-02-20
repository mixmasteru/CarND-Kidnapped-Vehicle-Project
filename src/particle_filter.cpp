#include "particle_filter.h"

#include <random>
#include <cmath>
#include <algorithm>
#include <iterator>
#include <string>
#include <vector>

#include "helper_functions.h"

using std::string;
using std::vector;
using std::normal_distribution;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
    /**
     * Set the number of particles. Initialize all particles to
     * first position (based on estimates of x, y, theta and their uncertainties
     * from GPS) and all weights to 1.
     * Add random Gaussian noise to each particle.
     *
     */
    num_particles = 100;  // Set the number of particles
    std::default_random_engine gen;

    // creates  normal (Gaussian) distribution for x
    normal_distribution<double> dist_x(x, std[0]);
    // create normal distributions for y and theta
    normal_distribution<double> dist_y(y, std[1]);
    normal_distribution<double> dist_theta(theta, std[2]);

    for (int id = 0; id < num_particles; ++id) {
        double sample_x, sample_y, sample_theta;

        // sample from these normal distributions
        sample_x = dist_x(gen);
        sample_y = dist_y(gen);
        sample_theta = dist_theta(gen);

        particles.push_back({id, sample_x, sample_y, sample_theta, 1.0});

    }

    is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[],
                                double velocity, double yaw_rate) {
    /**
     * Add measurements to each particle and add random Gaussian noise.
     * NOTE: When adding noise you may find std::normal_distribution
     *   and std::default_random_engine useful.
     *  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
     *  http://www.cplusplus.com/reference/random/default_random_engine/
     */
    std::default_random_engine gen;

    // Gaussian distribution for sensor noise
    normal_distribution<double> dist_x(0, std_pos[0]);
    normal_distribution<double> dist_y(0, std_pos[1]);
    normal_distribution<double> dist_theta(0, std_pos[2]);

    for (int i = 0; i < num_particles; i++) {
        if (fabs(yaw_rate) > 0.0001) {
            particles[i].x +=
                    (velocity / yaw_rate) * (sin(particles[i].theta + (yaw_rate * delta_t)) - sin(particles[i].theta));
            particles[i].y +=
                    (velocity / yaw_rate) * (cos(particles[i].theta) - cos(particles[i].theta + (yaw_rate * delta_t)));
            particles[i].theta += yaw_rate * delta_t;
        } else {
            particles[i].x += velocity * delta_t * cos(particles[i].theta);
            particles[i].y += velocity * delta_t * sin(particles[i].theta);
        }

        // add random Gaussian noise
        particles[i].x += dist_x(gen);
        particles[i].y += dist_y(gen);
        particles[i].theta += dist_theta(gen);
    }

}

void ParticleFilter::dataAssociation(const vector<LandmarkObs> &predicted,
                                     vector<LandmarkObs> &observations) {
    /**
     * Find the predicted measurement that is closest to each
     *   observed measurement and assign the observed measurement to this
     *   particular landmark.
     * NOTE: this method will NOT be called by the grading code. But you will
     *   probably find it useful to implement this method and use it as a helper
     *   during the updateWeights phase.
     */

    for (auto &observation : observations) {
        int nearest_landmark = 0;
        double min_distance = std::numeric_limits<float>::max();

        // check all predicted landmarks for nearest
        for (auto &predicted_landmark : predicted) {
            // calc Euclidean distance
            double distance = dist(predicted_landmark.x, predicted_landmark.y, observation.x, observation.y);
            // check min_dist and update if nearest
            if (distance < min_distance) {
                min_distance = distance;
                nearest_landmark = predicted_landmark.id;
            }
        }

        observation.id = nearest_landmark;
    }

}

void ParticleFilter::updateWeights(double sensor_range, const double std_landmark[],
                                   const vector<LandmarkObs> &observations,
                                   const Map &map_landmarks) {
    /**
     * Update the weights of each particle using a mult-variate Gaussian
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

    for (int i = 0; i < num_particles; i++) {

        // landmarks in the sensor range
        vector<LandmarkObs> landmarks;
        for (auto landmark : map_landmarks.landmark_list) {
            double distance = dist(landmark.x_f, landmark.y_f, particles[i].x, particles[i].y);
            if (distance <= sensor_range) {
                landmarks.push_back(LandmarkObs{landmark.id_i, landmark.x_f, landmark.y_f});
            }
        }

        // transform to map coordinates
        vector<LandmarkObs> map_observations;
        for (const auto &observation : observations) {
            LandmarkObs map_observation{};
            map_observation.id = observation.id;
            // to map x/y coordinate
            map_observation.x = particles[i].x + (cos(particles[i].theta) * observation.x) -
                                (sin(particles[i].theta) * observation.y);
            map_observation.y = particles[i].y + (sin(particles[i].theta) * observation.x) +
                                (cos(particles[i].theta) * observation.y);

            map_observations.push_back(map_observation);
        }

        dataAssociation(landmarks, map_observations);

        // calculate weights with gaussian distribution
        particles[i].weight = 1.0;

        double sig_x = std_landmark[0];
        double sig_y = std_landmark[1];
        double mu_x = 0;
        double mu_y = 0;

        for (auto &map_obs : map_observations) {
            unsigned int li = 0;
            do {
                if (landmarks[li].id == map_obs.id) {
                    mu_x = landmarks[li].x;
                    mu_y = landmarks[li].y;
                    break;
                }
                li++;
            } while (li < landmarks.size());

            // calc normalization
            double gauss_norm = 1 / (2 * M_PI * sig_x * sig_y);

            // calc exponent
            double exponent = (pow(map_obs.x - mu_x, 2) / (2 * pow(sig_x, 2)))
                              + (pow(map_obs.y - mu_y, 2) / (2 * pow(sig_y, 2)));

            // calc weight by normalization and exponent
            double weight = gauss_norm * exp(-exponent);

            particles[i].weight *= weight;
        }

    }

}

void ParticleFilter::resample() {
    /**
     * Resample particles with replacement with probability proportional
     *   to their weight.
     * NOTE: You may find std::discrete_distribution helpful here.
     *   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
     */

    std::default_random_engine gen;

    vector<Particle> resampled;

    vector<double> sample_weights;
    double max_weight = std::numeric_limits<double>::min();
    for (int i = 0; i < num_particles; i++) {
        sample_weights.push_back(particles[i].weight);
        if (particles[i].weight > max_weight) {
            max_weight = particles[i].weight;
        }
    }

    std::uniform_real_distribution<double> uniformRealDist(0, max_weight);
    std::uniform_int_distribution<int> uniformIntDist(0, num_particles - 1);
    int index = uniformIntDist(gen);

    double beta = 0.0;
    double mw = *std::max_element(std::begin(sample_weights), std::end(sample_weights));

    for (int i = 0; i < num_particles; i++) {
        beta += uniformRealDist(gen) * 2.0 * mw;

        while (beta > sample_weights[index]) {
            beta -= sample_weights[index];
            index = (index + 1) % num_particles;
        }

        resampled.push_back(particles[index]);
    }

    particles = resampled;

}

string ParticleFilter::getAssociations(const Particle &best) {
    vector<int> v = best.associations;
    std::stringstream ss;
    copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length() - 1);  // get rid of the trailing space
    return s;
}

string ParticleFilter::getSenseCoord(const Particle &best, const string &coord) {
    vector<double> v;

    if (coord == "X") {
        v = best.sense_x;
    } else {
        v = best.sense_y;
    }

    std::stringstream ss;
    copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length() - 1);  // get rid of the trailing space
    return s;
}