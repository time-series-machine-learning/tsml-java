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

/**This class defines the standard exponential distribution with rate parameter r*/
public class ExponentialDistribution extends GammaDistribution{
	//Parameter
	double rate;

	/**This general constructor creates a new exponential distribution with a
	specified rate*/
	public ExponentialDistribution(double r){
		setRate(r);
	}

	/**This default constructor creates a new exponential distribution with rate 1*/
	public ExponentialDistribution(){
		this(1);
	}

	/**This method sets the rate parameter*/
	public void setRate(double r){
		if (r <= 0) r = 1;
		rate = r;
		super.setParameters(1, 1 / rate);
	}

	/**This method gets the rate*/
	public double getRate(){
		return rate;
	}

	/**This method defines the getDensity function*/
	public double getDensity(double x){
		if (x < 0) return 0;
		else return rate * Math.exp(-rate * x);
	}

	/**This method defines the cumulative distribution function*/
	public double getCDF(double x){
		return 1 - Math.exp(- rate * x);
	}

	/**The method defines the getQuantile function*/
	public double getQuantile(double p){
		return -Math.log(1 - p) / rate;
	}
}

