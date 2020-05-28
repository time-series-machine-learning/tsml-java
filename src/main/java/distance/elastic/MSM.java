package distance.elastic;

import java.util.Random;
import core.contracts.Dataset;

/**
 * Some classes in this package may contain borrowed code from the timeseriesweka project (Bagnall, 2017), 
 * we might have modified (bug fixes, and improvements for efficiency) the original classes.
 * 
 */

public class MSM {
	
	
	public MSM() {
		
	}
	
	public double distance(double[] first, double[] second, double bsf, double c) {

		int m = first.length, n = second.length;
		int maxLength=(m>=n)?m:n;
		double[][]cost = MemorySpaceProvider.getInstance(maxLength).getDoubleMatrix();
//		double[][]cost = MemoryManager.getInstance().getDoubleMatrix(0);
		if (cost == null || cost.length < m || cost[0].length < n) {
			cost = new double[m][n];
		}

		// Initialization
		cost[0][0] = Math.abs(first[0] - second[0]);
		for (int i = 1; i < m; i++) {
			cost[i][0] = cost[i - 1][0] + calcualteCost(first[i], first[i - 1], second[0], c);
		}
		for (int i = 1; i < n; i++) {
			cost[0][i] = cost[0][i - 1] + calcualteCost(second[i], first[0], second[i - 1], c);
		}

		// Main Loop
		for (int i = 1; i < m; i++) {
			for (int j = 1; j < n; j++) {
				double d1, d2, d3;
				d1 = cost[i - 1][j - 1] + Math.abs(first[i] - second[j]);
				d2 = cost[i - 1][j] + calcualteCost(first[i], first[i - 1], second[j], c);
				d3 = cost[i][j - 1] + calcualteCost(second[j], first[i], second[j - 1], c);
				cost[i][j] = DistanceTools.Min3(d1, d2, d3);

			}
		}
		// Output
		double res = cost[m - 1][n - 1];
		MemorySpaceProvider.getInstance().returnDoubleMatrix(cost);
		return res;
	}

	private static final double calcualteCost(double new_point, double x, double y, double c) {

		double dist = 0;

		if (((x <= new_point) && (new_point <= y)) || ((y <= new_point) && (new_point <= x))) {
			dist = c;
		} else {
			dist = c + Math.min(Math.abs(new_point - x), Math.abs(new_point - y));
		}

		return dist;
	}
	
	public double get_random_cost(Dataset d, Random r) {
		return msmParams[r.nextInt(msmParams.length)];
	}

	protected static final double[] msmParams = { 0.01, 0.01375, 0.0175, 0.02125, 0.025, 0.02875, 0.0325, 0.03625, 0.04, 0.04375, 0.0475, 0.05125,
			0.055, 0.05875, 0.0625, 0.06625, 0.07, 0.07375, 0.0775, 0.08125, 0.085, 0.08875, 0.0925, 0.09625, 0.1, 0.136, 0.172, 0.208,
			0.244, 0.28, 0.316, 0.352, 0.388, 0.424, 0.46, 0.496, 0.532, 0.568, 0.604, 0.64, 0.676, 0.712, 0.748, 0.784, 0.82, 0.856,
			0.892, 0.928, 0.964, 1, 1.36, 1.72, 2.08, 2.44, 2.8, 3.16, 3.52, 3.88, 4.24, 4.6, 4.96, 5.32, 5.68, 6.04, 6.4, 6.76, 7.12,
			7.48, 7.84, 8.2, 8.56, 8.92, 9.28, 9.64, 10, 13.6, 17.2, 20.8, 24.4, 28, 31.6, 35.2, 38.8, 42.4, 46, 49.6, 53.2, 56.8, 60.4,
			64, 67.6, 71.2, 74.8, 78.4, 82, 85.6, 89.2, 92.8, 96.4, 100 };
}
