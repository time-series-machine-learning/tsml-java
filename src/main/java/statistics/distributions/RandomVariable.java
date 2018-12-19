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

public class RandomVariable{
	private Distribution distribution;
	private IntervalData intervalData;
	private String name;

	/**General constructor: create a new random variable with a specified
	probability distribution and name*/
	public RandomVariable(Distribution d, String n){
		distribution = d;
		name = n;
		intervalData = new IntervalData(distribution.getDomain(), name);

	}

	/**Special constructor: create a new random variable with a specified
	probability distribution and the name X*/
	public RandomVariable(Distribution d){
		this(d, "X");
	}

	/**Assign the probability distribution and create a corresponding data distribution*/
	public void setDistribution(Distribution d){
		distribution = d;
		intervalData.setDomain(distribution.getDomain());
	}

	/**Get the probability distribution*/
	public Distribution getDistribution(){
		return distribution;
	}

	/**Get the data distribution*/
	public IntervalData getIntervalData(){
		return intervalData;
	}

	/**Assign a value to the random variable*/
	public void setValue(double x){
		intervalData.setValue(x);
	}

	/**Get the current value of the random variable*/
	public double getValue(){
		return intervalData.getValue();
	}

	/**Simulate a value of the probability distribution and assign the value
	to the data distribution*/
	public void sample(){
		intervalData.setValue(distribution.simulate());
	}

	/**Simulate a value of the probability distribution, assign the value to the data distribution
	and return the value*/
	public double simulate(){
		double x = distribution.simulate();
		intervalData.setValue(x);
		return x;
	}

	/**Reset the data distribution*/
	public void reset(){
		intervalData.setDomain(distribution.getDomain());
	}

	/**Get the name of the random variable*/
	public String getName(){
		return name;
	}

	/**Assign a name to the random variable*/
	public void setName(String n){
		name = n;
		intervalData.setName(name);
	}
}

