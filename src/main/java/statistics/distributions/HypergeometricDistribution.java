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

/**This class models the hypergeometric distribution with parameters m (population size), n (sample
size), and r (number of type 1 objects)*/
public class HypergeometricDistribution extends Distribution{
	private int populationSize, sampleSize, type1Size;
	double c;

	/**General constructor: creates a new hypergeometric distribution with specified
	values of the parameters*/
	public HypergeometricDistribution(int m, int r, int n){
		setParameters(m, r, n);
	}

	/**Default constructor: creates a new hypergeometric distribuiton with
	parameters m = 100, r = 50, n = 10*/
	public HypergeometricDistribution(){
		this(100, 50, 10);
	}

	/**Set the parameters of the distribution*/
	public void setParameters(int m, int r, int n){
		//Correct for invalid parameters
		if (m < 1) m = 1;
		if (r < 0) r = 0; else if (r > m) r = m;
		if (n < 0) n = 0; else if (n > m) n = m;
		//Assign parameter values
		populationSize = m;
		type1Size = r;
		sampleSize = n;
		c = comb(populationSize, sampleSize);
		super.setParameters(Math.max(0, sampleSize - populationSize + type1Size), Math.min(type1Size, sampleSize), 1, DISCRETE);
	}

	/**Density function*/
	public double getDensity(double x){
		int k = (int)Math.rint(x);
		return comb(type1Size, k) * comb(populationSize - type1Size, sampleSize - k) / c;
	}

	/**Maximum value of the getDensity function*/
	public double getMaxDensity(){
		double mode = Math.floor(((double)(sampleSize + 1) * (type1Size + 1)) / (populationSize + 2));
		return getDensity(mode);
	}

	/**Mean*/
	public double getMean(){
		return (double)sampleSize * type1Size / populationSize;
	}

	/**Variance*/
	public double getVariance(){
		return (double)sampleSize * type1Size * (populationSize - type1Size) *
			(populationSize - sampleSize) / ( populationSize * populationSize * (populationSize - 1));
	}

	/**Set population size*/
	public void setPopulationSize(int m){
		setParameters(m, type1Size, sampleSize);
	}

	/**Get population size*/
	public int getPopulationSize(){
		return populationSize;
	}

	/**Set sub-population size*/
	public void setType1Size(int r){
		setParameters(populationSize, r, sampleSize);
	}

	/**Get sub-population size*/
	public int getType1Size(){
		return type1Size;
	}

	/**Set sample size*/
	public void setSampleSize(int n){
		setParameters(populationSize, type1Size, n);
	}

	/**Get sample size*/
	public int getSampleSize(){
		return sampleSize;
	}

	/**Simulate a value from the distribution*/
	public double simulate(){
		int j, k, u, m0;
		double x = 0;
		m0 = (int)populationSize;
		int[] b = new int[m0];
		for (int i = 0; i < m0; i++) b[i] = i;
		for (int i = 0; i < sampleSize; i++){
			k = m0 - i;
			u = (int)(k * Math.random());
			if (u < type1Size) x = x + 1;
			j = b[k - 1];
			b[k - 1] = b[u];
			b[u] = j;
		}
		return x;
	}
}

