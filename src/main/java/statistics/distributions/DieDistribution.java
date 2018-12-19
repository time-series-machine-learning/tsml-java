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

/**Distribution for a standard 6-sided die*/
public class DieDistribution extends FiniteDistribution{
	public final static int FAIR = 0, FLAT16 = 1, FLAT25 = 2, FLAT34 = 3, LEFT = 4, RIGHT = 5;

	/**General Constructor: creates a new die distribution with specified probabilities*/
	public DieDistribution(double[] p){
		super(1, 6, 1, p);
	}

	/**Special constructor: creates a new die distribution of a special type*/
	public DieDistribution(int n){
		super(1, 6, 1);
		setProbabilities(n);
	}

	/**Default constructor: creates a new fair die distribution*/
	public DieDistribution(){
		this(FAIR);
	}

	/**Specify probabilities of a special type*/
	public void setProbabilities(int n){
		if (n == FLAT16)
			setProbabilities(new double[] {1.0 / 4, 1.0 / 8, 1.0 / 8, 1.0 / 8, 1.0 / 8, 1.0 / 4});
		else if (n == FLAT25)
			setProbabilities(new double[] {1.0 / 8, 1.0 / 4, 1.0 / 8, 1.0 / 8, 1.0 / 4, 1.0 / 8});
		else if (n == FLAT34)
			setProbabilities(new double[] {1.0 / 8, 1.0 / 8, 1.0 / 4, 1.0 / 4, 1.0 / 8, 1.0 / 8});
		else if (n == LEFT)
			setProbabilities(new double[] {1.0 / 21, 2.0 / 21, 3.0 / 21, 4.0 / 21, 5.0 / 21, 6.0 / 21});
		else if (n == RIGHT)
			setProbabilities(new double[] {6.0 / 21, 5.0 / 21, 4.0 / 21, 3.0 / 21, 2.0 / 21, 1.0 / 21});
		else
			setProbabilities(new double[] {1.0 / 6, 1.0 / 6, 1.0 / 6, 1.0 / 6, 1.0 / 6, 1.0 / 6});
	}
}

