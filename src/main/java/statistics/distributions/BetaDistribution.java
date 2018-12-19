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

/**A Java implmentation of the beta distribution with specified left and right parameters*/
public class BetaDistribution extends Distribution{
	//Parameters
	private double left, right, c;

	/**General Constructor: creates a beta distribution with specified left and right
	parameters*/
	public BetaDistribution(double a, double b){
		setParameters(a, b);
	}

	/**Default constructor: creates a beta distribution with left and right parameters
	equal to 1*/
	public BetaDistribution(){
		this(1, 1);
	}

	/**Set the parameters, compute the normalizing constant c, and specifies the
	interval and partition*/
	public void setParameters(double a, double b){
		double lower, upper, step;
		//Correct parameters that are out of bounds
		if (a <= 0) a = 1;
		if (b <= 0) b = 1;
		//Assign parameters
		left = a; right = b;
		//Compute the normalizing constant
		c = logGamma(left + right) - logGamma(left) - logGamma(right);
		//Specifiy the interval and partiton
		super.setParameters(0, 1, 0.001, CONTINUOUS);
	}

	/**Sets the left parameter*/
	public void setLeft(double a){
		setParameters(a, right);
	}

	/**Sets the right parameter*/
	public void setRight(double b){
		setParameters(left, b);
	}

	/**Get the left paramter*/
	public double getLeft(){
		return left;
	}

	/**Get the right parameter*/
	public double getRight(){
		return right;
	}

  /**Define the beta getDensity function*/
	public double getDensity(double x){
		if ((x < 0) | (x > 1)) return 0;
		else if ((x == 0) & (left == 1)) return right;
		else if ((x == 0) & (left < 1)) return Double.POSITIVE_INFINITY;
		else if ((x == 0) & (left > 1)) return 0;
		else if ((x == 1) & (right == 1)) return left;
		else if ((x == 1) & (right < 1)) return Double.POSITIVE_INFINITY;
		else if ((x == 1) & (right > 1)) return 0;
		else return Math.exp(c + (left - 1) * Math.log(x) + (right - 1) * Math.log(1 - x));
	}

	/**Compute the maximum getDensity*/
	public double getMaxDensity(){
		double mode;
		if (left < 1) mode = 0.01;
		else if (right <= 1) mode = 0.99;
		else mode = (left - 1) / (left + right - 2);
		return getDensity(mode);
	}

	/**Compute the mean in closed form*/
	public double getMean(){
		return left / (left + right);
	}

	/**Compute the variance in closed form*/
	public double getVariance(){
		return left * right / ((left + right) * (left + right) * (left + right + 1));
	}

	/**Compute the cumulative distribution function. The beta CDF is built into
	the superclass Distribution*/
	public double getCDF(double x){
		return betaCDF(x, left, right);
	}
}

