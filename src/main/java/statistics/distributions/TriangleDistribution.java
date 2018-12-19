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

/**This class models the triangle distribution on a specified interval.  If (X, Y) is
uniformly distributed on a triangular region, then X and Y have triangular distribuitons.*/
public class TriangleDistribution extends Distribution{
	private int orientation;
	private double c, minValue, maxValue;
	public final static int UP = 0,  DOWN = 1;

	/**This general constructor creates a new triangle distribution on a specified
	interval and with a specified orientation.*/
	public TriangleDistribution(double a, double b, int i){
		setParameters(a, b, i);
	}

	/**This default constructor creates a new triangle distribution on the interval
	(0, 1) with positive slope*/
	public TriangleDistribution(){
		this(0, 1, UP);
	}

	/**This method sets the parameters: the minimum value, maximum value, and
	orientation.*/
	public void setParameters(double a, double b, int i){
		minValue = a;
		maxValue = b;
		orientation = i;
		double stepSize = (maxValue - minValue) / 100;
		super.setParameters(minValue, maxValue, stepSize, CONTINUOUS);
		//Compute normalizing constant
		c = (maxValue - minValue) * (maxValue - minValue);
	}

	//**This method computes the density.*/
	public double getDensity(double x){
		if (minValue <= x & x <= maxValue){
			if (orientation == UP) return 2 * (x - minValue) / c;
			else return 2 * (maxValue - x) / c;
		}
		else return 0;
	}

	/**This method computes the maximum value of the getDensity function.*/
	public double getMaxDensity(){
		double mode;
		if (orientation == UP) mode = maxValue;
		else mode = minValue;
		return getDensity(mode);
	}

	/**This method computes the mean.*/
	public double getMean(){
		if (orientation == UP) return minValue / 3 + 2 * maxValue / 3;
		else return 2 * minValue / 3 + maxValue / 3;
	}

	/**This method computes the variance.*/
	public double getVariance(){
		return (maxValue - minValue) * (maxValue - minValue) / 18;
	}

	/**This method returns the minimum value.*/
	public double getMinValue(){
		return minValue;
	}

	/**This method returns the maximum value.*/
	public double getMaxValue(){
		return maxValue;
	}

	/**This method returns the orientation.*/
	public int getOrientation(){
		return orientation;
	}

	/**This method simulates a value from the distribution.*/
	public double simulate(){
		double u = minValue + (maxValue - minValue) * Math.random();
		double v = minValue + (maxValue - minValue) * Math.random();
		if (orientation == UP) return Math.max(u, v);
		else return Math.min(u, v);
	}

	/**This method computes the cumulative distribution function.*/
	public double getCDF(double x){
		if (orientation == UP) return (x - minValue) * (x - minValue) / c;
		else return 1 - (maxValue - x) * (maxValue - x) / c;
	}
}

