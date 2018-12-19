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

/**This class models the Weibull distribution with specified shape and scale
parameters*/
public class WeibullDistribution extends Distribution{
	//Variables
	double shape, scale, c;

	/**This general constructor creates a new Weibull distribution with spcified
	shape and scale parameters*/
	public WeibullDistribution(double k, double b){
		setParameters(k, b);
	}

	/**This default constructor creates a new Weibull distribution with shape
	parameter 1 and scale parameter 1*/
	public WeibullDistribution(){
		this(1, 1);
	}

	/**This method sets the shape and scale parameter. The normalizing constant
	is computed and the default interval defined*/
	public void setParameters(double k, double b){
		double upper, width;
		if (k <= 0) k = 1;
		if (b <= 0) b = 1;
		//Assign parameters
		shape = k; scale = b;
		//Compute normalizing constant
		c = shape / Math.pow(scale, shape);
		//Define interval
		upper = Math.ceil(getMean() + 4 * getSD());
		width = upper/ 100;
		super.setParameters(0, upper, width, CONTINUOUS);
	}

	/**This method compues teh denstiy function*/
	public double getDensity(double x){
		return c * Math.pow(x, shape - 1) * Math.exp(-Math.pow(x / scale, shape));
	}

	/**This method returns the maximum value of the getDensity function*/
	public double getMaxDensity(){
		double mode;
		if (shape < 1) mode = getDomain().getLowerValue();
		else mode = scale * Math.pow((shape - 1) / shape, 1 / shape);
		return getDensity(mode);
	}

	/**The method returns the mean*/
	public double getMean(){
		return scale * gamma(1 + 1 / shape);
	}

	/**This method returns the variance*/
	public double getVariance(){
		double mu = getMean();
		return scale * scale * gamma(1 + 2 / shape) - mu * mu;
	}

	/**This method computes the cumulative distribution function*/
	public double getCDF(double x){
		return 1 - Math.exp(-Math.pow(x / scale, shape));
	}

	/**This method returns the getQuantile function*/
	public double getQuantile(double p){
		return scale * Math.pow(-Math.log(1 - p), 1 / shape);
	}

	/**This method computes the failure rate function*/
	public double getFailureRate(double x){
		return shape * Math.pow(x, shape - 1) / Math.pow(scale, shape);
	}


	/**This method returns the shape parameter*/
	public double getShape(){
		return shape;
	}

	/**This method sets the shape parameter*/
	public void setShape(double k){
		setParameters(k, scale);
	}

	/**This method returns the scale parameter*/
	public double getScale(){
		return scale;
	}

	/**This method sets the shape parameter*/
	public void setScale(double b){
		setParameters(shape, b);
	}
}

