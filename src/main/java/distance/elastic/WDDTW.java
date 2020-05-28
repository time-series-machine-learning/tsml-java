package distance.elastic;

/**
 * Some classes in this package may contain borrowed code from the timeseriesweka project (Bagnall, 2017), 
 * we might have modified (bug fixes, and improvements for efficiency) the original classes.
 * 
 */

public class WDDTW extends WDTW{
	protected double[] deriv1, deriv2;

	public synchronized double distance(double[] first, double[] second, double bsf, double g) {
		if (deriv1 == null || deriv1.length != first.length) {
			deriv1 = new double[first.length];
		}
		DDTW.getDeriv(deriv1, first);

		if (deriv2 == null || deriv2.length != second.length) {
			deriv2 = new double[second.length];
		}
		DDTW.getDeriv(deriv2, second);
		return super.distance(deriv1, deriv2, bsf, g);
	}	
}
