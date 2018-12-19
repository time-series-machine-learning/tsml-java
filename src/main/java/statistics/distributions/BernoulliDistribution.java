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

/**The Bernoulli distribution with parameter p*/
public class BernoulliDistribution extends BinomialDistribution{

	/**This general constructor creates a new Bernoulli distribution with a specified parameter*/
	public BernoulliDistribution(double p){
		super(1, p);
	}

	/**This default constructor creates a new Bernoulli distribution with parameter p = 0.5*/
	public BernoulliDistribution(){
		this(0.5);
	}

	/**This method overrides the corresponding method in BinomialDistribution so that the number of trials 1 cannot be changed*/
	public void setTrials(int n){
		super.setTrials(1);
	}

	/**This method returns the maximum value of the getDensity function*/
	public double getMaxDensity(){
		return 1;
	}

}

