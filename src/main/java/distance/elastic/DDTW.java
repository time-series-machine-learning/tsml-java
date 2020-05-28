package distance.elastic;

/**
 * Some classes in this package may contain borrowed code from the timeseriesweka project (Bagnall, 2017), 
 * we might have modified (bug fixes, and improvements for efficiency) the original classes.
 * 
 */

public class DDTW extends DTW{
	double[] deriv1, deriv2;

	public DDTW() {
		
	}
	
	public synchronized double distance(double[] series1, double[] series2, double bsf, int w) {
//		System.out.println("calling ddtw with w="+w);

		if (deriv1 == null || deriv1.length != series1.length) {
			deriv1 = new double[series1.length];
		}
		getDeriv(deriv1,series1);
		
		if (deriv2 == null || deriv2.length != series2.length) {
			deriv2 = new double[series2.length];
		}
		getDeriv(deriv2,series2);

		return super.distance(deriv1, deriv2, bsf,w);
	}
	
	protected static final void getDeriv(double[]d,double[] series) {
		for (int i = 1; i < series.length - 1 ; i++) { 
			d[i] = ((series[i] - series[i - 1]) + ((series[i + 1] - series[i - 1]) / 2.0)) / 2.0;
		}

		d[0] = d[1];
		d[d.length - 1] = d[d.length - 2];
		
	}
	
}
