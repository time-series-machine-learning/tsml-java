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

/**The binomial distribution with specified parameters:
the number of trials and the probability of success*/
public class BinomialDistribution extends Distribution{
	//Variables
	private int trials;
	private double probability;

	/**General constructor: creates the binomial distribution with specified
	parameters*/
	public BinomialDistribution(int n, double p){
		setParameters(n, p);
	}

	/**Default constructor: creates the binomial distribution with 10 trials
	and probability of success 1/2*/
	public BinomialDistribution(){
		this(10, 0.5);
	}

	/**Set the parameters*/
	public void setParameters(int n, double p){
		//Correct invalid parameters
		if (n < 1) n = 1;
		if (p < 0) p = 0; else if (p > 1) p = 1;
		trials = n; probability = p;
		super.setParameters(0, trials, 1, DISCRETE);
	}

	/**Set the number of trails*/
	public void setTrials(int n){
		setParameters(n, probability);
	}

	/**Get the number of trials*/
	public int getTrials(){
		return trials;
	}

	/**Set the probability of success*/
	public void setProbability(double p){
		setParameters(trials, p);
	}

	/**Get the probability of success*/
	public double getProbability(){
		return probability;
	}

	/**Define the binomial getDensity function*/
	public double getDensity(double x){
		int k = (int)Math.rint(x);
		if (k < 0 | k > trials) return 0;
		if (probability == 0){
			if (k == 0) return 1;
			else return 0;
		}
		else if (probability == 1){
			if (k == trials) return 1;
			else return 0;
		}
		else return comb(trials, k) * Math.pow(probability, k) * Math.pow( 1 - probability, trials - k);
	}

	/**Specify the maximum getDensity*/
	public double getMaxDensity(){
	double mode = Math.min(Math.floor((trials + 1) * probability), trials);
		return getDensity(mode);
	}

	/**Give the mean in closed form*/
	public double getMean(){
		return trials * probability;
	}

	/**Specify the variance in close form*/
	public double getVariance(){
		return trials * probability * (1 - probability);
	}

	/**Specify the CDF in terms of the beta CDF*/
	public double getCDF(double x){
		if (x < 0) return 0;
		else if (x >= trials) return 1;
		else return 1 - betaCDF(probability, x + 1, trials - x);
	}

	/**Simulate the binomial distribution as the number of successes in n trials*/
	public double simulate(){
		int successes = 0;
		for (int i = 1; i <= trials; i++){
			if (Math.random() < probability) successes++;
		}
		return successes;
	}
}

