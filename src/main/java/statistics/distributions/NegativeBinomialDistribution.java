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

/**This class models the negative binomial distribution with specified successes
parameter and probability parameter.*/
public class NegativeBinomialDistribution extends Distribution{
	//Paramters
	private int successes;
	private double probability;

	/**General Constructor: creates a new negative binomial distribution with
	given parameter values.*/
	public NegativeBinomialDistribution(int k, double p){
		setParameters(k, p);
	}

	/**Default Constructor: creates a new negative binomial distribution with
	successes parameter 1 and probability parameter 0.5,*/
	public NegativeBinomialDistribution(){
		this(1, 0.5);
	}

	/**This method set the paramters and the set of values.*/
	public void setParameters(int k, double p){
		//Correct for invalid parameters
		if(k < 1) k = 1;
		if(p <= 0) p = 0.05;
		if(p > 1) p = 1;
		//Assign parameters
		successes = k;
		probability = p;
		//Set truncated values
		super.setParameters(successes, Math.ceil(getMean() + 4 * getSD()), 1, DISCRETE);
	}

	/**Set the successes parameters*/
	public void setSuccesses(int k){
		setParameters(k, probability);
	}

	/**Get the successes parameter*/
	public int getSuccesses(){
		return successes;
	}

	/**Get the probability parameter*/
	public double getProbability(){
		return probability;
	}

	/**Set the probability parameters*/
	public void setProbability(double p){
		setParameters(successes, p);
	}

	/**Density function*/
	public double getDensity(double x){
		int n = (int)Math.rint(x);
		if(n < successes) return 0;
		else return comb(n - 1, successes - 1) * Math.pow(probability, successes)
			* Math.pow(1 - probability, n - successes);
	}

	/**Maximum value of getDensity function*/
	public double getMaxDensity(){
		double mode = (successes - 1) / probability + 1;
		return getDensity(mode);
	}

	/**Mean*/
	public double getMean(){
		return successes / probability;
	}

	/**Variance*/
	public double getVariance(){
		return (successes * (1 - probability)) / (probability * probability);
	}

	/**Simulate a value*/
	public double simulate(){
		int count = 0, trials = 0;
		while (count < successes){
			if (Math.random() < probability) count++;
			trials++;
		}
		System.out.println("In simulate, prob ="+probability+"\t success= "+successes+"\t trials ="+trials);
		
		return trials;
	}
}

