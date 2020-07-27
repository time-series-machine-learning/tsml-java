package distance.elastic;

import static distance.elastic.DistanceTools.sim;
import java.util.Random;
import core.contracts.Dataset;

/**
 * Some classes in this package may contain borrowed code from the timeseriesweka project (Bagnall, 2017), 
 * we might have modified (bug fixes, and improvements for efficiency) the original classes.
 * 
 */

public class LCSS {
	
	public LCSS() {
		
	}
	
	public synchronized double distance(double[] series1, double[] series2, double bsf, int windowSize, double epsilon) {
		if (windowSize == -1) {
			windowSize = series1.length;
		}

		int length1 = series1.length;
		int length2 = series2.length;

		int maxLength = Math.max(length1, length2);
		int minLength = Math.min(length1, length2);

		int [][]matrix = MemorySpaceProvider.getInstance(maxLength).getIntMatrix();
//		int[][] matrix = MemoryManager.getInstance().getIntMatrix(0);

		int i, j;
	
		matrix[0][0] = sim(series1[0], series2[0], epsilon);
		for (i = 1; i < Math.min(length1, 1 + windowSize); i++) {
			matrix[i][0] = (sim(series1[i], series2[0], epsilon)==1)?sim(series1[i], series2[0], epsilon):matrix[i-1][0];
		}

		for (j = 1; j < Math.min(length2, 1 + windowSize); j++) {
			matrix[0][j] = (sim(series1[0], series2[j], epsilon)==1?sim(series1[0], series2[j], epsilon):matrix[0][j-1]);
		}
		
		if (j < length2)
			matrix[0][j] = Integer.MIN_VALUE;


		for (i = 1; i < length1; i++) {
			int jStart = (i - windowSize < 1) ? 1 : i - windowSize;
			int jStop = (i + windowSize + 1 > length2) ? length2 : i + windowSize + 1;
			
			if (i-windowSize-1>=0)
				matrix[i][i-windowSize-1] = Integer.MIN_VALUE;
			for (j = jStart; j < jStop; j++) {
				if (sim(series1[i], series2[j], epsilon) == 1) {
					matrix[i][j] = matrix[i - 1][j - 1] + 1;
				} else {
					matrix[i][j] = max(matrix[i - 1][j - 1], matrix[i][j - 1], matrix[i - 1][j]);
				}
			}
			if (jStop < length2)
				matrix[i][jStop] = Integer.MIN_VALUE;
		}
		
		double res = 1.0 - 1.0 * matrix[length1 - 1][length2 - 1] / minLength;
		MemorySpaceProvider.getInstance().returnIntMatrix(matrix);
		return res;

	}
	
	
	public static final int max(int A, int B, int C) {
		if (A > B) {
			if (A > C) {
				return A;
			} else {
				// C > A > B
				return C;
			}
		} else {
			if (B > C) {
				// B > A and B > C
				return B;
			} else {
				// C > B > A
				return C;
			}
		}
	}	
	
	public int get_random_window(Dataset d, Random r) {
//		int x = (d.length() +1) / 4;
		return r.nextInt((d.length() +1) / 4); //TODO
	} 	
	
	public double get_random_epsilon(Dataset d, Random r) {
		double stdTrain = DistanceTools.stdv_p(d);
		double stdFloor = stdTrain * 0.2;
		double e = r.nextDouble()*(stdTrain-stdFloor)+stdFloor;
		return e;
	} 
	
}
