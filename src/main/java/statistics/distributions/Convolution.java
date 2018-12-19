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

/**This class creates the n-fold convolution of a given distribution*/
public class Convolution extends Distribution{
	private Distribution distribution;
	private int power;
	private double[][] pdf;

	/**This general constructor: creates a new convolution distribution corresponding
	to a specified distribution and convolution power*/
	public Convolution(Distribution d, int n){
		setParameters(d, n);
	}

	/**This defalut constructor creates a new convolution distribution corrrepsonding to the
	uniform distribution on (0,1), with convolution power 5.*/
	public Convolution(){
		this(new ContinuousUniformDistribution(0, 1), 5);
	}

	/**This method sets the parameters: the distribution and convolution power. The method computes
	and store pdf values*/
	public void setParameters(Distribution d, int n){
		//Correct for invalid parameters
		if (n < 1) n = 1;
		distribution = d; power = n;
		Domain domain = distribution.getDomain();
		double l = domain.getLowerValue(), u = domain.getUpperValue(), w = domain.getWidth(), p, dx;
		int t = distribution.getType();
		if (t == DISCRETE) dx = 1; else dx = w;
		super.setParameters(power * l, power * u, w, t);
		int m = domain.getSize();
		pdf = new double[power][];
		for (int k = 0; k < n; k++) pdf[k] = new double[(k + 1) * m - k];
			for (int j = 0; j < m; j++) pdf[0][j] = distribution.getDensity(domain.getValue(j));
				for (int k = 1; k < n; k++){
				for (int j = 0; j < (k + 1) * m - k; j++){
					p = 0;
					for (int i = Math.max(0, j - m + 1); i < Math.min(j+1, k * m - k + 1); i++){
					   p = p + pdf[k - 1][i] * pdf[0][j - i];
				}
				pdf[k][j] = p;
			}
		}
	}

	/**Density function*/
	public double getDensity(double x){
		return pdf[power - 1][getDomain().getIndex(x)];
	}

	/**Mean*/
	public double getMean(){
		return power * distribution.getMean();
	}

	/**Variance*/
	public double getVariance(){
		return power * distribution.getVariance();
	}

	/**Simulate a value from the distribution*/
	public double simulate(){
		double sum = 0;
		for (int i = 0; i < power; i++)	sum = sum + distribution.simulate();
		return sum;
	}

	/**This method sets the convolution power.*/
	public void setPower(int n){
		setParameters(distribution, n);
	}

	/**This method returns the convolution power.*/
	public int getPower(){
		return power;
	}

	/**This method sets the distribution.*/
	public void setDistribution(Distribution d){
		setParameters(d, power);
	}

	/**This method returns the distribution.*/
	public Distribution getDistribution(){
		return distribution;
	}
}



