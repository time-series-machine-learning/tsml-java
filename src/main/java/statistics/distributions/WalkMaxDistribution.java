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

/**This class models the distribution of the maximum value of a symmetric random walk on the interval
[0, n].*/
public class WalkMaxDistribution extends Distribution{
	//Paramters
	private int steps;

	/**This general constructor creates a new max walk distribution with a specified time parameter.*/
	public WalkMaxDistribution(int n){
		setSteps(n);
	}

	/**This default constructor creates a new walk max distribution with time parameter 10.*/
	public WalkMaxDistribution(){
		this(10);
	}

	/**This method sets the time parameter.*/
	public void setSteps(int n){
		if (n < 1) n = 1;
		steps = n;
		super.setParameters(0, steps, 1, DISCRETE);
	}

	/**This method defines the density function.*/
	public double getDensity(double x){
		int k = (int)Math.rint(x), m;
		if ((k + steps) % 2 == 0) m = (k + steps) / 2;
		else m = (k + steps + 1) / 2;
		return comb(steps, m) / Math.pow(2 , steps);
	}

	/**This method returns the maximum value of the density function.*/
	public double getMaxDensity(){
		return getDensity(0);
	}

	/**This method returns the number ofsteps.*/
	public double getSteps(){
		return steps;
	}

	/**This method simulates a value from the distribution.*/
	public double simulate(){
		int step, max = 0, position = 0;
		for (int i = 1; i <= steps; i++){
			if (Math.random() < 0.5) step = 1;
			else step = -1;
			position = position + step;
			if (position > max) max = position;
		}
		return max;
	}
}

