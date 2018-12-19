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

/**This class models the lognormal distribution with specified parameters*/
public class LogNormalDistribution extends Distribution{
	//variables
	public final static double C = Math.sqrt(2 * Math.PI);
	private double mu, sigma;

	/**This general constructor creates a new lognormal distribution with
	specified parameters*/
	public LogNormalDistribution(double m, double s){
		setParameters(m, s);
	}

	/**This default constructor creates the standard lognormal distribution*/
	public LogNormalDistribution(){
		this(0, 1);
	}

	/**This method sets the parameters, computes the default interval*/
	public void setParameters(double m, double s){
		if (s <= 0) s = 1;
		mu = m; sigma = s;
		double upper = getMean() + 3 * getSD();
		super.setParameters(0, upper, 0.01 * upper, CONTINUOUS);
	}

	/**This method computes the getDensity function*/
	public double getDensity(double x){
		double z = (Math.log(x) - mu) / sigma;
		return Math.exp(- z * z / 2) / (x * C * sigma);
	}

	/**This method computes the maximum value of the getDensity function*/
	public double getMaxDensity(){
		double mode = Math.exp(mu - sigma * sigma);
		return getDensity(mode);
	}

	/**This method computes the mean*/
	public double getMean(){
		return Math.exp(mu + sigma * sigma / 2);
	}

	/**This method computes the variance*/
	public double getVariance(){
		double a = mu + sigma * sigma;
		return Math.exp(2 * a) - Math.exp(mu + a);
	}

	/**This method simulates a value from the distribution*/
	public double simulate(){
		double r = Math.sqrt(-2 * Math.log(Math.random()));
		double theta = 2 * Math.PI * Math.random();
		return Math.exp(mu + sigma * r * Math.cos(theta));
	}

	/**This method returns mu*/
	public double getMu(){
		return mu;
	}

	/**This method sets mu*/
	public void setMu(double m){
		setParameters(m, sigma);
	}

	/**This method gets sigma*/
	public double getSigma(){
		return sigma;
	}

	/**This method sets sigma*/
	public void setSigma(double s){
		setParameters(mu, s);
	}

	/**This method computes the cumulative distribution function*/
	public double getCDF(double x){
		double z = (Math.log(x) - mu) / sigma;
		if (z >= 0) return 0.5 + 0.5 * gammaCDF(z * z / 2, 0.5);
		else return 0.5 - 0.5 * gammaCDF(z * z / 2, 0.5);
  }
}

