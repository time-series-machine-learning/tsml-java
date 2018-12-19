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

/** This class models the Fisher F distribution with a spcified number of
degrees of freedom in the numerator and denominator*/
public class FisherDistribution extends Distribution{
	private int nDegrees, dDegrees;
	private double c;

	/**This general constructor creates a new Fisher distribution with a
	specified number of degrees of freedom in the numerator and denominator*/
	public FisherDistribution(int n, int d){
		setParameters(n, d);
	}

	/**This default constructor creates a new Fisher distribution with 5
	degrees of freedom in numerator and denominator*/
	public FisherDistribution(){
		this(5, 5);
	}

	/**This method sets the parameters, the degrees of freedom in the numerator
	and denominator. Additionally, the normalizing constant and default interval
	are computed*/
	public void setParameters(int n, int d){
		double upper, width;
		//Correct invalid parameters
		if (n < 1) n = 1; if (d < 1) d = 1;
		nDegrees = n; dDegrees = d;
		//Compute normalizing constant
		c = logGamma(0.5 * (nDegrees + dDegrees)) - logGamma(0.5 * nDegrees)
			- logGamma(0.5 * dDegrees) + 0.5 * nDegrees * (Math.log(nDegrees)
			- Math.log(dDegrees));
		//Compute interval
		if (dDegrees <= 4) upper = 20; else upper = getMean() + 4 * getSD();
		width = 0.01 * upper;
		super.setParameters(0, upper, width, CONTINUOUS);
	}

	/**This method computes the denisty function*/
	public double getDensity(double x){
		if (x < 0) return 0;
		else if (x == 0 & nDegrees == 1) return Double.POSITIVE_INFINITY;
		else return Math.exp(c + (0.5 * nDegrees - 1) * Math.log(x)
			- 0.5 * (nDegrees + dDegrees) * Math.log(1 + nDegrees * x / dDegrees));
	}

	/**This method computes the maximum value of the getDensity function*/
	public double getMaxDensity(){
		double mode;
		if (nDegrees == 1) mode = getDomain().getLowerValue();
		else mode = (double)((nDegrees - 2) * dDegrees) / (nDegrees * (dDegrees + 2));
		return getDensity(mode);
	}

	/**This method returns the mean*/
	public double getMean(){
		if (dDegrees <= 2) return Double.POSITIVE_INFINITY;
		else return (double)dDegrees / (dDegrees - 2);
	}

	/**This method returns the variance*/
	public double getVariance(){
		if (dDegrees <= 2) return Double.NaN;
		else if (dDegrees <= 4) return Double.POSITIVE_INFINITY;
		else return 2.0 * (dDegrees / (dDegrees - 2)) * (dDegrees / (dDegrees - 2))
			* (dDegrees + nDegrees - 2) / (nDegrees * (dDegrees - 4));
	}

	/**This method computes the cumulative distribution function in terms of
	the beta CDF*/
	public double getCDF(double x){
		double u = dDegrees / (dDegrees + nDegrees * x);
		if (x < 0) return 0;
		else return 1 - betaCDF(u, 0.5 * dDegrees, 0.5 * nDegrees);
	}

	/**This method returns the numerator degrees of freedom*/
	public double getNDegrees(){
		return nDegrees;
	}

	/**This method sets the numerator degrees of freedom*/
	public void setNDegrees(int n){
		setParameters(n, dDegrees);
	}

	/**This method gets the denominator degrees of freedom*/
	public double getDDegrees(){
		return dDegrees;
	}

	/**This method sets the denominator degrees of freedom*/
	public void setDDegrees(int d){
		setParameters(nDegrees, d);
	}

	/**This method simulates a value from the distribution*/
	public double simulate(){
		double U, V, Z, r, theta;
		U = 0;
		for (int i = 1; i <= dDegrees; i++){
			r = Math.sqrt(-2 * Math.log(Math.random()));
			theta = 2 * Math.PI * Math.random();
			Z = r * Math.cos(theta);
			U = U + Z * Z;
		}
		V = 0;
		for (int j = 1; j <= dDegrees; j++){
			r = Math.sqrt(-2 * Math.log(Math.random()));
			theta = 2 * Math.PI * Math.random();
			Z = r * Math.cos(theta);
			V = V + Z * Z;
		}
		return (U / nDegrees) / (V / dDegrees);
	}
}

