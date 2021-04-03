package tsml.classifiers.shapelet_based.distances;

import tsml.classifiers.shapelet_based.type.ShapeletMV;
import tsml.data_containers.TimeSeriesInstance;

public class ShapeletDistanceEuclidean implements ShapeletDistanceFunction {
    @Override
    public double calculate(ShapeletMV shapelet, TimeSeriesInstance instance) {
        double bestSum = Double.MAX_VALUE;
        double sum;
        double[] subseq;
        double temp;
        int shapeletLength = shapelet.getLength();
        //m-l+1
        //multivariate instances that are split dont have a class value on them.
        int minLength = instance.getMinLength();

        for (int i = 0; i < minLength - shapeletLength + 1; i++)
        {
            sum = shapelet.getDistanceToInstance(i,instance);

            if (sum < bestSum)
            {
                bestSum = sum;
                //System.out.println(i);
            }
        }

        double dist = (bestSum == 0.0) ? 0.0 : (1.0 / shapeletLength * bestSum);
        return dist;
    }


}
