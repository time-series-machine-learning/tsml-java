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

/**The geometric distribution with parameter p*/
public class GeometricDistribution extends NegativeBinomialDistribution{

	/**General Constructor: creates a new geometric distribution with parameter p*/
	public GeometricDistribution(double p){
		super(1, p);
	}

	/**Default Constructor: creates a new geometric distribution with parameter 0.5*/
	public GeometricDistribution(){
		this(0.5);
	}

	/**Override set parameters*/
	public void setParameters(int k, double p){
		super.setParameters(1, p);
	}

	/**Override set successes*/
	public void setSuccesses(int k){}


}

