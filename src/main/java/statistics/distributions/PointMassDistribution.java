// Point mass at x0
package statistics.distributions;

public class PointMassDistribution extends Distribution{
	//Paramter
	double x0;

	//Constructor
	public PointMassDistribution(double x0){
		setParameters(x0);
	}

	public PointMassDistribution(){
		this(0);
	}

	public void setParameters(double x0){
		this.x0 = x0;
		super.setParameters(x0, x0, 1, DISCRETE);
	}

	public double getDensity(double x){
		if (x == x0) return 1;
		else return 0;
	}

	public double getMaxDensity(){
		return 1;
	}

	public double getMean(){
		return x0;
	}

	public double getVariance(){
		return 0;
	}

	public double getParameter(int i){
		return x0;
	}

	public double simulate(){
		return x0;
	}

	public double getQuantile(double p){
		return x0;
	}

	public double CDF(double x){
		if (x < x0) return 0;
		else return 1;
	}

	public String name(){
		return "Point Mass Distribution";
	}
}

