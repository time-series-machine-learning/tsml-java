package distance.elastic;

import java.util.List;

import core.contracts.Dataset;

/**
 * Some classes in this package may contain borrowed code from the timeseriesweka project (Bagnall, 2017), 
 * we might have modified (bug fixes, and improvements for efficiency) the original classes.
 * 
 */

public class DistanceTools {
	public static int[] getInclusive10(int min, int max){
	        int[] output = new int[10];

	        double diff = (double)(max-min)/9;
	        double[] doubleOut = new double[10];
	        doubleOut[0] = min;
	        output[0] = min;
	        for(int i = 1; i < 9; i++){
	            doubleOut[i] = doubleOut[i-1]+diff;
	            output[i] = (int)Math.round(doubleOut[i]);
	        }
	        output[9] = max; // to make sure max isn't omitted due to double imprecision
	        return output;
	    }

	    public static double[] getInclusive10(double min, double max){
	        double[] output = new double[10];
	        double diff = (max-min)/9;
	        output[0] = min;
	        for(int i = 1; i < 9; i++){
	            output[i] = output[i-1]+diff;
	        }
	        output[9] = max;

	        return output;
	    }
	    public static int sim(double a, double b, double epsilon) {
			return (Math.abs(a - b) <= epsilon) ? 1 : 0;
		}

		public static double stdv_p(Dataset train) {

			double sumx = 0;
			double sumx2 = 0;
			double[] ins2array;
			for (int i = 0; i < train.size(); i++) {
				ins2array = train.get_series(i);
				for (int j = 0; j < ins2array.length; j++) {
										// avoid
										// classVal
					sumx += ins2array[j];
					sumx2 += ins2array[j] * ins2array[j];
				}
			}
			int n = train.size() * (train.length());
			double mean = sumx / n;
			return Math.sqrt(sumx2 / (n) - mean * mean);  //TODO check this?? possible issue here 

		}	    
	    
		public static double stdv_p(List<double[]> train) {

			double sumx = 0;
			double sumx2 = 0;
			double[] ins2array;
			for (int i = 0; i < train.size(); i++) {
				ins2array = train.get(i);
				for (int j = 0; j < ins2array.length; j++) {
										// avoid
										// classVal
					sumx += ins2array[j];
					sumx2 += ins2array[j] * ins2array[j];
				}
			}
			int n = train.size() * (train.get(0).length);
			double mean = sumx / n;
			return Math.sqrt(sumx2 / (n) - mean * mean); //TODO check this?? possible issue here 

		}
		
//		public static <C,D> double stdv_p(ArrayDataset<C,D> train) {
//
//			double sumx = 0;
//			double sumx2 = 0;
//			double[] ins2array;
//			for (int i = 0; i < train.size(); i++) {
//				ins2array = train.getPrimitiveDoubleArray(i);
//				for (int j = 0; j < ins2array.length - 1; j++) {// -1 to
//										// avoid
//										// classVal
//					sumx += ins2array[j];
//					sumx2 += ins2array[j] * ins2array[j];
//				}
//			}
//			int n = train.size() * (train.get(0).size());
//			double mean = sumx / n;
//			return Math.sqrt(sumx2 / (n) - mean * mean);
//
//		}		
		
		public final static double Min3(final double a, final double b, final double c) {
			return (a <= b) ? ((a <= c) ? a : c) : (b <= c) ? b : c;
		}

		public static int ArgMin3(final double a, final double b, final double c) {
			return (a <= b) ? ((a <= c) ? 0 : 2) : (b <= c) ? 1 : 2;
		}
		
		public static double sum(final double... tab) {
			double res = 0.0;
			for (double d : tab)
				res += d;
			return res;
		}
		
		public static double max(final double... tab) {
			double max = Double.NEGATIVE_INFINITY;
			for (double d : tab){
				if(max<d){
					max = d;
				}
			}
			return max;
		}
		
		public static double min(final double... tab) {
			double min = Double.POSITIVE_INFINITY;
			for (double d : tab){
				if(d<min){
					min = d;
				}
			}
			return min;
		}
}
