// Mixture
package statistics.distributions;

public class MixtureDistribution extends Distribution{
	Distribution[] dist;
	int n, type;
	double minValue, maxValue, lowerValue, upperValue, stepSize;
	double[] prob;

	//Constructors
	public MixtureDistribution(Distribution[] d, double[] p){
		setParameters(d, p);
	}

	public MixtureDistribution(Distribution d0, Distribution d1, double a){
		setParameters(d0, d1, a);
	}

	public void setParameters(Distribution[] d, double[] p){
		double minLower = Double.POSITIVE_INFINITY, maxUpper = Double.NEGATIVE_INFINITY, minWidth = Double.POSITIVE_INFINITY;
		double a, b, w;
		dist = d;
		prob = p;
		int t0 = dist[0].getType(), t;
		n = dist.length;
		boolean mixed = false;
		for (int i = 0; i < n; i++){
			t = dist[i].getType();
			if (t == DISCRETE) a = dist[i].getDomain().getLowerValue(); else a = dist[i].getDomain().getLowerBound();
			if (a < minLower) minLower = a;
			if (t == DISCRETE) b = dist[i].getDomain().getUpperValue(); else b = dist[i].getDomain().getUpperBound();
			if (b > maxUpper) maxUpper = b;
			w = dist[i].getDomain().getWidth();
			if (w < minWidth) minWidth = w;
			if (t != t0) mixed = true;
		}
		if (mixed) t = 2; else t = t0;
		super.setParameters(minLower, maxUpper, minWidth, t);
	}

	public void setParameters(Distribution d0, Distribution d1, double a){
		setParameters(new Distribution[]{d0, d1}, new double[]{1 - a, a});
	}

	//Density
	public double getDensity(double x){
		double d = 0;
		for (int i = 0; i < n; i++) d = d + prob[i] * dist[i].getDensity(x);
		return d;
	}

	//Mean
	public double getMean(){
		double sum = 0;
		for (int i = 0; i < n; i++) sum = sum + prob[i] * dist[i].getMean();
		return sum;
	}

	//Variance
	public double getVariance(){
		double sum = 0, mu = getMean(), m;
		for (int i = 0; i < n; i++){
			m = dist[i].getMean();
			sum = sum + prob[i] * (dist[i].getVariance() + m * m);
		}
		return sum - mu * mu;
	}

	//Simulate
	public double simulate(){
		double sum = 0, p = Math.random();
		int i = -1;
		while (sum < p & i < n){
			sum = sum + prob[i];
			i = i + 1;
		}
		return dist[i].simulate();
	}
}

