package utilities;

public class NumUtils {

    public static boolean isPercentage(double value) {
        return value >= 0 && value <= 1;
    }

    public static boolean isNearlyEqual(double a, double b, double eps){
        return Math.abs(a - b) < eps;
    }

	public static boolean isNearlyEqual(double a, double b){
	    return isNearlyEqual(a,b, 1e-6);
	}


}
