package timeseriesweka.filters;

import java.util.function.Function;

public class DerivativeCache extends Cache<double[], double[]> {
    public DerivativeCache() {
        super(DerivativeCache::getDerivative);
    }


    public static double[] getDerivative(double[] input){
        return DerivativeFilter.getDerivative(input, false);
    }
}
