package distance.elastic;

import java.util.Random;
import core.contracts.Dataset;

/**
 * Some classes in this package may contain borrowed code from the timeseriesweka project (Bagnall, 2017), 
 * we might have modified (bug fixes, and improvements for efficiency) the original classes.
 * 
 */

public class ERP {

	public ERP() {
		
	}
	
	double[] curr, prev;

	public synchronized double distance(double[] first, double[] second, double bsf, int windowSize, double gValue) {
		// base case - we're assuming class val is last. If this is
		// true, this method is fine,
		// if not, we'll default to the DTW class

		int m = first.length;
		int n = second.length;

		if (curr == null || curr.length < m) {
			curr = new double[m];
			prev = new double[m];
		} else {
			// FPH: init to 0 just in case, didn't check if
			// important
			for (int i = 0; i < curr.length; i++) {
				curr[i] = 0.0;
				prev[i] = 0.0;
			}
		}

		// size of edit distance band
		// bandsize is the maximum allowed distance to the diagonal
		// int band = (int) Math.ceil(v2.getDimensionality() *
		// bandSize);
//		int band = (int) Math.ceil(m * bandSize);
		int band = windowSize;

		// g parameter for local usage
		for (int i = 0; i < m; i++) {
			// Swap current and prev arrays. We'll just overwrite
			// the new curr.
			{
				double[] temp = prev;
				prev = curr;
				curr = temp;
			}
			int l = i - (band + 1);
			if (l < 0) {
				l = 0;
			}
			int r = i + (band + 1);
			if (r > (m - 1)) {
				r = (m - 1);
			}

			for (int j = l; j <= r; j++) {
				if (Math.abs(i - j) <= band) {
					// compute squared distance of feature
					// vectors
					double val1 = first[i];
					double val2 = gValue;
					double diff = (val1 - val2);
//					final double d1 = Math.sqrt(diff * diff);
					final double d1 = diff;//FPH simplificaiton

					val1 = gValue;
					val2 = second[j];
					diff = (val1 - val2);
//					final double d2 = Math.sqrt(diff * diff);
					final double d2 = diff;

					val1 = first[i];
					val2 = second[j];
					diff = (val1 - val2);
//					final double d12 = Math.sqrt(diff * diff);
					final double d12 = diff;

					final double dist1 = d1 * d1;
					final double dist2 = d2 * d2;
					final double dist12 = d12 * d12;

					final double cost;

					if ((i + j) != 0) {
						if ((i == 0) || ((j != 0) && (((prev[j - 1] + dist12) > (curr[j - 1] + dist2))
										&& ((curr[j - 1] + dist2) < (prev[j] + dist1))))) {
							// del
							cost = curr[j - 1] + dist2;
						} else if ((j == 0) || ((i != 0) && (((prev[j - 1] + dist12) > (prev[j] + dist1))
										&& ((prev[j] + dist1) < (curr[j - 1] + dist2))))) {
							// ins
							cost = prev[j] + dist1;
						} else {
							// match
							cost = prev[j - 1] + dist12;
						}
					} else {
						cost = 0;
					}

					curr[j] = cost;
					// steps[i][j] = step;
				} else {
					curr[j] = Double.POSITIVE_INFINITY; // outside
									    // band
				}
			}
		}

		return Math.sqrt(curr[m - 1]);	//TODO do we need sqrt here
	}
	
	public int get_random_window(Dataset d, Random r) {
//		int x = (d.length() +1) / 4;
		int w = r.nextInt(d.length()/ 4+1);
		return w;
	} 	
	
	public double get_random_g(Dataset d, Random r) {
		double stdv = DistanceTools.stdv_p(d);		
		double g = r.nextDouble()*.8*stdv+0.2*stdv; //[0.2*stdv,stdv]
		return g;
	} 		
}
