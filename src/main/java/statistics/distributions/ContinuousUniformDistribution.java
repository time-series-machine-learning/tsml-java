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

/**This class models the uniform distribution on a specified interval.*/
public class ContinuousUniformDistribution extends Distribution{
	private double minValue, maxValue;

	/**This general constructor creates a new uniform distribution on a specified interval.*/
	public ContinuousUniformDistribution(double a, double b){
		setParameters(a, b);
	}

	/**This default constructor creates a new uniform distribuiton on (0, 1).*/
	public ContinuousUniformDistribution(){
		this(0, 1);
	}

	/**This method sets the parameters: the minimum and maximum values of the interval.*/
	public void setParameters(double a, double b){
		minValue = a; maxValue = b;
		double step = 0.01 * (maxValue - minValue);
		super.setParameters(minValue, maxValue, step, CONTINUOUS);
	}

	/**This method computes the density function.*/
	public double getDensity(double x){
		if (minValue <= x & x <= maxValue) return 1 / (maxValue - minValue);
		else return 0;
	}

	/**This method computes the maximum value of the getDensity function.*/
	public double getMaxDensity(){
		return 1 / (maxValue - minValue);
	}

	/**This method computes the mean.*/
	public double getMean(){
		return (minValue + maxValue) / 2;
	}

	/**This method computes the variance.*/
	public double getVariance(){
		return (maxValue - minValue) * (maxValue - minValue) / 12;
	}

	/**This method computes the cumulative distribution function.*/
	public double getCDF(double x){
		if (x < minValue) return 0;
		else if (x >= maxValue) return 1;
		else return (x - minValue) / (maxValue - minValue);
	}

	/**This method computes the getQuantile function.*/
	public double getQuantile(double p){
		if (p < 0) p = 0; else if (p > 1) p = 1;
		return minValue + (maxValue - minValue) * p;
	}

	/**This method gets the minimum value.*/
	public double getMinValue(){
		return minValue;
	}

	/**This method returns the maximum value.*/
	public double getMaxValue(){
		return maxValue;
	}

	/**This method simulates a value from the distribution.*/
	public double simulate(){
		return minValue + Math.random() * (maxValue - minValue);
	}
}

