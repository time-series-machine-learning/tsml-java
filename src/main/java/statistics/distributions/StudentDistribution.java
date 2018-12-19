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

/**This class models the student t distribution with a specifed degrees of freeom
parameter*/
public class StudentDistribution extends Distribution{
	private int degrees;
	private double c;

	/**This general constructor creates a new student distribution with a specified
	degrees of freedom*/
	public StudentDistribution(int n){
		setDegrees(n);
	}

	/**This default constructor creates a new student distribuion with 1 degree
	of freedom*/
	public StudentDistribution(){
		this(1);
	}

	/**This method sets the degrees of freedom*/
	public void setDegrees(int n){
		//Correct invalid parameter
		if (n < 1) n = 1;
		//Assign parameter
		degrees = n;
		//Compute normalizing constant
		c = logGamma(0.5 * (degrees + 1)) - 0.5 * Math.log(degrees) - 0.5 * Math.log(Math.PI) - logGamma(0.5 * degrees);
		//Compute upper bound
		double upper;
		if (n == 1) upper = 8;
		else if (n == 2) upper = 7;
		else upper = Math.ceil(getMean() + 4 * getSD());
		super.setParameters(-upper, upper, upper / 50, CONTINUOUS);
	}

	/**This method computes the getDensity function*/
	public double getDensity(double x){
		return Math.exp(c - 0.5 * (degrees + 1) * Math.log(1 + x * x / degrees));
	}

	/**This method returns the maximum value of the getDensity function*/
	public double getMaxDensity(){
		return getDensity(0);
	}

	/**This method returns the mean*/
	public double getMean(){
		if (degrees == 1) return Double.NaN;
		else return 0;
	}

	/**This method returns the variance*/
	public double getVariance(){
		if (degrees == 1) return Double.NaN;
		else if (degrees == 2) return Double.POSITIVE_INFINITY;
		else return (double)degrees / (degrees - 2);
	}

	/**This method computes the cumulative distribution function in terms of the
	beta CDF*/
	public double getCDF(double x){
		double u = degrees / (degrees + x * x);
		if (x > 0) return 1 - 0.5 * betaCDF(u, 0.5 * degrees, 0.5);
		else return 0.5 * betaCDF(u, 0.5 * degrees, 0.5);
	}

	/**This method returns the degrees of freedom*/
	public double getDegrees(){
		return degrees;
	}

	/**This method simulates a value of the distribution*/
	public double simulate(){
		double v, z, r, theta;
		v = 0;
		for (int i = 1; i <= degrees; i++){
			r = Math.sqrt(-2 * Math.log(Math.random()));
			theta = 2 * Math.PI * Math.random();
			z = r * Math.cos(theta);
			v = v + z * z;
		}
		r = Math.sqrt(-2 * Math.log(Math.random()));
		theta = 2 * Math.PI * Math.random();
		z = r * Math.cos(theta);
		return z / Math.sqrt(v / degrees);
	}
}

