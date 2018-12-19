/*
Copyright (C) 2001  Kyle Siegrist, Dawn Duehring

This program is free software; you can redistribute it and/or modify it
under the terms of the GNU General Public License as published by the Free
Software Foundation; either version 2 of the License, or (at your option)
any later version.

This program is distributed in the hope that it will be useful, but without
any warranty; without even the implied warranty of merchantability or
fitness for a particular purpose. See the GNU General Public License for
more details. You should have received a copy of the GNU General Public
License along with this program; if not, write to the Free Software
Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA 02111-1307, USA.
*/
package statistics.distributions;

/**The binomial distribution with a random number of trials*/
public class BinomialRandomNDistribution extends Distribution{
	//Variables
	double probability, sum;
	Distribution dist;

	/**This general constructor creates a new randomized binomial distribution with a specified probability of success and a specified distribution for
	the number of trials*/
	public BinomialRandomNDistribution(Distribution d, double p){
		setParameters(d, p);
	}

	/**Special constructor: creates a new randomized binomial distribution with a specified probability of success and the uniform distribution on {1, 2, 3, 4, 5, 6} for the number of trials*/
	public BinomialRandomNDistribution(double p){
		this(new DiscreteUniformDistribution(1, 6, 1), p);
	}

	/**This default constructor: creates a new randomized binomial distribution with probability of success 0.5 and the uniform distribution on {1, 2, 3, 4, 5, 6} for the number of trials*/
	public BinomialRandomNDistribution(){
		this(new DiscreteUniformDistribution(1, 6, 1), 0.5);
	}

	/**Set the parameters: the distribution for the number of trials and the
	probability of success*/
	public void setParameters(Distribution d, double p){
		dist = d;
		probability = p;
		super.setParameters(0, dist.getDomain().getUpperValue(), 1, DISCRETE);
	}

	//Density
	public double getDensity(double x){
		int k = (int)Math.rint(x);
		double trials;
		if (probability == 0){
			if (k == 0) return 1;
			else return 0;
		}
		else if (probability == 1) return dist.getDensity(k);
		else{
			sum = 0;
			for(int i = 0; i < dist.getDomain().getSize(); i++){
				trials = dist.getDomain().getValue(i);
				sum = sum + dist.getDensity(trials) *
					comb(trials, k) * Math.pow(probability, k) * Math.pow(1 - probability, trials - k);
			}
			return sum;
		}
	}

	public double getMean(){
		return dist.getMean() * probability;
	}

	public double getVariance(){
		return dist.getMean() * probability * (1 - probability) + dist.getVariance() * probability * probability;
	}

	public double simulate(){
		int trials = (int)dist.simulate();
		int successes = 0;
		for (int i = 1; i <= trials; i++){
			if (Math.random() < probability) successes++;
		}
		return successes;
	}

}



