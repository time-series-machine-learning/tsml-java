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

/**Gamma distribution with a specified shape parameter and scale parameter*/
public class GammaDistribution extends Distribution{
	//Parameters
	private double shape, scale, c;

	/**General Constructor: creates a new gamma distribution with shape parameter
	k and scale parameter b*/
	public GammaDistribution(double k, double b){
		setParameters(k, b);
	}

	/**Default Constructor: creates a new gamma distribution with shape parameter
	1 and scale parameter 1*/
	public GammaDistribution(){
		this(1, 1);
	}

	/**Set parameters and assign the default partition*/
	public void setParameters(double k, double b){
		double upperBound;
		//Correct invalid parameters
		if(k < 0) k = 1;
		if(b < 0) b = 1;
		shape = k;
		scale = b;
		//Normalizing constant
		c = shape * Math.log(scale) + logGamma(shape);
		//Assign default partition:
		upperBound = getMean() + 4 * getSD();
		super.setParameters(0, upperBound, 0.01 * upperBound, CONTINUOUS);
	}

	/** Get shape parameters*/
	public double getShape(){
		return shape;
	}

	/** Get scale parameters*/
	public double getScale(){
		return scale;
	}

	/**Density function */
	public double getDensity(double x){
		if (x < 0) return 0;
		else if (x == 0 & shape < 1) return Double.POSITIVE_INFINITY;
		else if (x == 0 & shape == 1) return Math.exp(-c);
		else if (x == 0 & shape > 1) return 0;
		else return Math.exp(-c + (shape - 1) * Math.log(x) - x / scale);
	}

	/** Maximum value of getDensity function*/
	public double getMaxDensity(){
		double mode;
		if (shape < 1) mode = 0.01; else mode = scale * (shape - 1);
		return getDensity(mode);
	}

	/** Mean */
	public double getMean(){
		return shape * scale;
	}

	/**Variance*/
	public double getVariance(){
		return shape * scale * scale;
	}

	/** Cumulative distribution function*/
	public double getCDF(double x){
		return gammaCDF(x / scale, shape);
	}

	/** Simulate a value*/
	public double simulate(){
		/*If shape parameter k is an integer, simulate as the k'th arrival time
		in a Poisson proccess
		*/
		if (shape == Math.rint(shape)){
			double sum = 0;
			for (int i = 1; i <= shape; i++){
				sum = sum - scale * Math.log(1 - Math.random());
			}
			return sum;
		}
		else return super.simulate();
	}
}

