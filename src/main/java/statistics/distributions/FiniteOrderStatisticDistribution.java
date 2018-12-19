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

/** This class models the distribution of the k'th order statistic for a sample of size n
chosen without replacement from {1, 2, ..., N} .*/
public class FiniteOrderStatisticDistribution extends Distribution{
	Distribution dist;
	private int sampleSize, populationSize, order;

	/**This general constructor creates a new finite order statistic distribution with specified
	population and sample sizes, and specified order.*/
	public FiniteOrderStatisticDistribution(int N, int n, int k){
		setParameters(N, n, k);
	}

	/**This default constructor creates a new finite order statistic distribution with population
	size 50, sample size 10, and order 5.*/
	public FiniteOrderStatisticDistribution(){
		this(50, 10, 5);
	}

	/**This method sets the parameters: the sample size, population size, and order.*/
	public void setParameters(int N, int n, int k){
		populationSize = N;
		sampleSize = n;
		order = k;
		super.setParameters(order, populationSize - sampleSize + order, 1, Distribution.DISCRETE);
	}

	/**This method computes the getDensity.*/
	public double getDensity(double x){
		int i = (int)Math.rint(x);
		return comb(i - 1, order - 1)
			* comb(populationSize - i, sampleSize - order) / comb(populationSize, sampleSize);
	}

	/**This method computes the mean.*/
	public double getMean(){
		return (double)order * (populationSize + 1) / (sampleSize + 1);
	}


	/**This method computes the variance.*/
	public double getVariance(){
		return (double)(populationSize + 1) * (populationSize - sampleSize)
			* order * (sampleSize + 1 - order) / ((sampleSize + 1) * (sampleSize + 1) * (sampleSize + 2));
	}

	/**This method sets the population size.*/
	public void setPopulationSize(int N){
		setParameters(N, sampleSize, order);
	}

	/**This method returns the population size.*/
	public int getPopulationSize(){
 		return populationSize;
	}

	/**This method sets the sample size.*/
	public void setSampleSize(int n){
		setParameters(populationSize, n, order);
	}

	/**This method returns the sampleSize.*/
	public int getSampleSize(){
		return sampleSize;
	}

	/**This method sets the order.*/
	public void setOrder(int k){
		setParameters(populationSize, sampleSize, k);
	}

	/**This method returns the order.*/
	public int getOrder(){
		return order;
	}
}

