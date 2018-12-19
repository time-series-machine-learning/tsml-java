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

/**This class models the Cauchy distribution*/
public class CauchyDistribution extends StudentDistribution{
	//Constructor
	public CauchyDistribution(){
		super(1);
	}

	/**This method sets the degrees of freedom to 1.*/
	public void setDegrees(int n){
		super.setDegrees(1);
	}

	/**This method computes the CDF. This overrides the corresponding method in StudentDistribution.*/
	public double getCDF(double x){
		return 0.5 + Math.atan(x) / Math.PI;
	}

	/**This method computes the quantile function. This overrides
	the corresponding method in StudentDistribution.*/
	public double getQuantile(double p){
		return Math.tan(Math.PI * (p - 0.5));
	}
}

