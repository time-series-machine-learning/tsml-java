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

/**This class models the distribution of the number of distinct sample values
when a sample of a specified size is chosen with replacement from a finite
population of a specified size*/
public class BirthdayDistribution extends Distribution{
	private int popSize, sampleSize;
	private double[][] prob;

	/**This general constructor creates a new birthday distribution with
	a specified population size and sample size*/
	public BirthdayDistribution(int n, int k){
		setParameters(n, k);
	}

	/**This default constructor creates a new birthday distribution with
	population size 365 and sample size 20*/
	public BirthdayDistribution(){
		this(365, 20);
	}

	/**This method sets the parameters: the population size and the sample size.
	Also, the probabilities are computed and stored in an array*/
	public void setParameters(int n, int k){
		//Correct for invalid parameters
		if (n < 1) n = 1;
		if (k < 1) k = 1;
		int upperIndex;
		popSize = n; sampleSize = k;
		super.setParameters(1, Math.min(popSize, sampleSize), 1, DISCRETE);
		prob = new double[sampleSize + 1][popSize + 1];
		prob[0][0] = 1; prob[1][1] = 1;
		for (int j = 1; j < sampleSize; j++){
			if (j < popSize) upperIndex = j + 1; else upperIndex = (int)popSize;
			for (int m = 1; m <= upperIndex; m++){
				prob[j+1][m] = prob[j][m] * ((double)m / popSize)
					+ prob[j][m - 1] * ((double)(popSize - m + 1) / popSize);
			}
		}
	}

	/**This method computes the getDensity function*/
	public double getDensity(double x){
		int m = (int)(Math.rint(x));
		return prob[sampleSize][m];
	}

	/**This method computes the mean*/
	public double getMean(){
		return popSize * (1 - Math.pow(1 - 1.0 / popSize, sampleSize));
	}

	/**This method computes the variance*/
	public double getVariance(){
		return popSize * (popSize - 1) * Math.pow(1 - 2.0 / popSize, sampleSize)
			+ popSize * Math.pow(1 - 1.0 / popSize, sampleSize)
			- popSize * popSize * Math.pow(1 - 1.0 / popSize, 2 * sampleSize);
	}

	/**This method returns the population size*/
	public double getPopSize(){
		return popSize;
	}

	/**This method sets the population size*/
	public void setPopSize(int n){
		setParameters(n, sampleSize);
	}

	/**This method returns the sample size*/
	public double getSampleSize(){
		return sampleSize;
	}

	/**This method sets the sample size*/
	public void setSampleSize(int k){
		setParameters(popSize, k);
	}

	/**This method simulates a value from the distribution, as the number
	of distinct sample values*/
	public double simulate(){
		int[] count = new int[popSize];
		double distinct = 0;
		for (int i = 1; i <= sampleSize; i++){
			int j = (int)(popSize * Math.random());
			if (count[j] == 0) distinct++;
			count[j] = count[j]++;
		}
		return distinct;
	}
}

