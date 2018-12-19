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

/**This class models the distribution of the position at time n for a random walk
on the interval [0, n].*/
package statistics.distributions;

public class WalkPositionDistribution extends Distribution{
	//Paramters
	private int steps ;
	private double probability;

	/**This general constructor creates a new distribution with specified time and
	probability parameters.*/
	public WalkPositionDistribution(int n, double p){
		setParameters(n, p);
	}

	/**This default constructor creates a new WalkPositionDistribution with time parameter 10
	and probability p.*/
	public WalkPositionDistribution(){
		this(10, 0.5);
	}

	/**This method sets the time and probability parameters.*/
	public void setParameters(int n, double p){
		if (n < 0) n = 0;
		if (p < 0) p = 0; else if (p > 1) p = 1;
		steps = n;
		probability = p;
		super.setParameters(-steps, steps, 2, DISCRETE);
	}

	/**This method computes the density function.*/
	public double getDensity(double x){
		int k = (int)Math.rint(x), m = (k + steps) / 2;
		return comb(steps, m) * Math.pow(probability, m) * Math.pow(1 - probability, steps - m);
	}

	/**This method returns the maximum value of the density function.*/
	public double getMaxDensity(){
		double mode = 2 * Math.min(Math.floor((steps + 1) * probability), steps) - steps;
		return getDensity(mode);
	}

	/**This method computes the mean.*/
	public double getMean(){
		return 2 * steps * probability - steps;
	}

	/**This method computes the variance.*/
	public double getVariance(){
		return 4 * steps * probability * (1 - probability);
	}

	/**This method returns the number of steps.*/
	public double getSteps(){
		return steps;
	}

	/**This method returns the probability of a step to the right.*/
	public double getProbability(){
		return probability;
	}

	/**This method simulates a value from the distribution.*/
	public double simulate(){
		int step, position = 0;
		for (int i = 1; i <= steps; i++){
			if (Math.random() < probability) step = 1;
			else step = -1;
			position = position + step;
		}
		return position;
	}
}

