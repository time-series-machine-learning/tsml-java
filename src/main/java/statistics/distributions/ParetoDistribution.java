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

/**This class models the Pareto distribution with a specified parameter*/
public class ParetoDistribution extends Distribution{
	//Variable
	private double parameter;

	/**This general constructor creates a new Pareto distribuiton with a
	specified parameter*/
	public ParetoDistribution(double a){
		setParameter(a);
	}

	/**The default constructor creates a new Pareto distribution with parameter 1*/
	public ParetoDistribution(){
		this(1);
	}

	/**This method sets the parameter and computes the default interval*/
	public void setParameter(double a){
		if (a <= 0) a = 1;
		parameter = a;
		double upper = 20 / parameter;
		double width = (upper - 1) / 100;
		super.setParameters(1, upper, width, CONTINUOUS);
	}

	/**This method returns the parameter*/
	public double getParameter(){
		return parameter;
	}

	/**This method computes the getDensity function*/
	public double getDensity(double x){
		if (x < 1) return 0;
		else return parameter / Math.pow(x, parameter + 1);
	}

	/**This method returns the maximum value of the getDensity function*/
	public double getMaxDensity(){
		return parameter;
	}

	/**This method computes the mean*/
	public double getMean(){
		if (parameter > 1) return parameter / (parameter - 1);
		else return Double.POSITIVE_INFINITY;
	}

	/**This method computes the variance*/
	public double getVariance(){
		if (parameter > 2) return parameter / ((parameter - 1) * (parameter - 1) * (parameter - 2));
		else if (parameter > 1) return Double.POSITIVE_INFINITY;
		else return Double.NaN;
	}

	/**This method comptues the cumulative distribution function*/
	public double getCDF(double x){
		return 1 - Math.pow(1 / x, parameter);
	}

	/**This method computes the getQuantile function*/
	public double getQuantile(double p){
		return 1 / Math.pow(1 - p, 1 / parameter);
	}
}

