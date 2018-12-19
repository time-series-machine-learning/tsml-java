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

/**The Poisson distribution with a specified rate parameter*/
public class PoissonDistribution extends Distribution{
	//Variables
	double parameter;

	/**Default constructor: creates a new Poisson distribution with a given
	parameter value*/
	public PoissonDistribution(double r){
		setParameter(r);
	}

	/**Default constructor: creates a new Poisson distribtiton with parameter 1*/
	public PoissonDistribution(){
		this(1);
	}

	/**Sets the parameter*/
	public void setParameter(double r){
		//Correct for invalid parameter:
		if(r < 0) r = 1;
		parameter = r;
		//Sets the truncated set of values
		double a = Math.ceil(getMean() - 4 * getSD()), b = Math.ceil(getMean() + 4 * getSD());
		if (a < 0) a = 0;
		super.setParameters(a, b, 1, DISCRETE);
	}

	/**Parameter*/
	public double getParameter(){
		return parameter;
	}

	/**Density function*/
	public double getDensity(double x){
		int k = (int)Math.rint(x);
		if(k < 0) return 0;
		else return Math.exp(-parameter) * (Math.pow(parameter, k) / factorial(k));
	}

	/**Maximum value of the getDensity function*/
	public double getMaxDensity(){
		double mode = Math.floor(parameter);
		return getDensity(mode);
	}

	/**Cumulative distribution function*/
	public double getCDF(double x){
		return 1 - gammaCDF(parameter, x + 1);
	}

	/**Mean*/
	public double getMean(){
		return parameter;
	}

	/**Variance*/
	public double getVariance(){
		return parameter;
	}

	/**Simulate a value*/
	public double simulate(){
		int arrivals = 0;
		double sum = -Math.log(1 - Math.random());
		while (sum <= parameter){
			arrivals++;
			sum = sum - Math.log(1 - Math.random());
		}
		return arrivals;
	}
}

