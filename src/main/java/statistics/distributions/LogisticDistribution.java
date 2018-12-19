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

/**This class models the logistic distribution*/
public class LogisticDistribution extends Distribution{

	/**This default constructor creates a new logsitic distribution*/
	public LogisticDistribution(){
		super.setParameters(-7, 7, 0.14, CONTINUOUS);
	}

	/**This method computes the getDensity function*/
	public double getDensity(double x){
	double e = Math.exp(x);
		return e / ((1 + e)*(1 + e));
	}

	 /**This method computes the maximum value of the getDensity function*/
	 public double getMaxDensity(){
		return 0.25;
	}

	/**This method computes the cumulative distribution function*/
	public double getCDF(double x){
		double e = Math.exp(x);
		return e / (1 + e);
	}

	/**This method comptues the getQuantile function*/
	public double getQuantile(double p){
		return Math.log(p / (1 - p));
	}

	/**This method returns the mean*/
	public double getMean(){
		return 0;
	}

	/**This method computes the variance*/
	public double getVariance(){
		return Math.PI * Math.PI / 3;
	}
}

