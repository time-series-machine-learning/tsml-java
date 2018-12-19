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

/**This class encapsulates the normal distribution with specified parameters*/
public class NormalDistribution extends Distribution{
	//Paramters
	public final static double C = Math.sqrt(2 * Math.PI);
	private double mu, sigma, cSigma;

	/**This general constructor creates a new normal distribution with specified
	parameter values*/
	public NormalDistribution(double mu, double sigma){
		setParameters(mu, sigma);
	}

	/**This default constructor creates a new standard normal distribution*/
	public NormalDistribution(){
		this(0, 1);
	}

	/**This method sets the parameters*/
	public void setParameters(double m, double s){
		double lower, upper, width;
		//Correct for invalid sigma
		if (s < 0) s = 1;
		mu = m; sigma = s;
		cSigma = C * sigma;
		upper = mu + 4 * sigma;
		lower = mu - 4 * sigma;
		width = (upper - lower) / 100;
		super.setParameters(lower, upper, width, CONTINUOUS);
	}

	/**This method defines the getDensity function*/
	public double getDensity(double x){
		double z = (x - mu) / sigma;
		return Math.exp(- z * z / 2) / cSigma;
	}

	/**This method returns the maximum value of the getDensity function*/
	public double getMaxDensity(){
		return getDensity(mu);
	}

	/**This method returns the median*/
	public double getMedian(){
		return mu;
	}

	/**This method returns the mean*/
	public double getMean(){
		return mu;
	}

	/**This method returns the variance*/
	public double getVariance(){
		return sigma * sigma;
	}

	/**This method simulates a value from the distribution*/
	public double simulate(){
		double r = Math.sqrt(-2 * Math.log(RNG.nextDouble()));
		double theta = 2 * Math.PI * RNG.nextDouble();
		return mu + sigma * r * Math.cos(theta);
	}

	/**This method returns the location parameter*/
	public double getMu(){
		return mu;
	}

	/**This method sets the location parameter*/
	public void setMu(double m){
		setParameters(m, sigma);
	}

	/**This method gets the scale parameter*/
	public double getSigma(){
		return sigma;
	}

	/**This method sets the scale parameter*/
	public void setSigma(double s){
		setParameters(mu, s);
	}

	/**This method computes the cumulative distribution function*/
	public double getCDF(double x){
		double z = (x - mu) / sigma;
		if (z >= 0) return 0.5 + 0.5 * gammaCDF(z * z / 2, 0.5);
		else return 0.5 - 0.5 * gammaCDF(z * z / 2, 0.5);
	}
	public String toString()
	{
		String str=super.toString();
		str+=" mean "+mu+" sigma = "+sigma;
		return str;
	}
}

