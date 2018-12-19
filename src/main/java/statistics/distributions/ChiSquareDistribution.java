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

/**This class defines the chi-square distribution with a specifed degrees of
freedom*/
public class ChiSquareDistribution extends GammaDistribution{
	int degrees;

	/**This general constructor creates a new chi-square distribuiton with a
	specified degrees of freedom parameter*/
	public ChiSquareDistribution(int n){
		setDegrees(n);
	}

	public ChiSquareDistribution(){
		this(1);
	}

	/**This method sets the degrees of freedom*/
	public void setDegrees(int n){
		//Correct invalid parameter
		if (n <= 0) n = 1;
		degrees = n;
		super.setParameters(0.5 * degrees, 2);
	}

	/**This method returns the degrees of freedom*/
	public int getDegrees(){
		return degrees;
	}

	/**This method simulates a value from the distribuiton, as the sum of squares
	of independent, standard normal distribution*/
	public double simulate(){
		double V, Z, r, theta;
		V = 0;
		for (int i = 1; i <= degrees; i++){
			r = Math.sqrt(-2 * Math.log(Math.random()));
			theta = 2 * Math.PI * Math.random();
			Z = r * Math.cos(theta);
			V = V + Z * Z;
		}
		return V;
	}
}

