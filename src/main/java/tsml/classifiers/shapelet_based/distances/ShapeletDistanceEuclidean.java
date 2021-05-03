package tsml.classifiers.shapelet_based.distances;

import tsml.classifiers.shapelet_based.type.ShapeletIndependentMV;
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


    public static void main(String[] args){

        double[][] data1 = {
                {0.1,0.9,0.7,0.3,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1},
                {0.1,0.1,0.1,0.1,0.1,0.1,0.3,0.5,0.7,0.1,0.1,0.1,0.1},
                {0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1}
        };

        double[][] data2 = {
                {0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.9,0.1,0.1,0.1,0.1},
                {0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1},
                {0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1}
        };

        double[][] data3 = {
                {1.1,2.1,3.1,4.1,5.1,6.1,7.1,8.1,9.1,10.1,11.1,12.1,13.1},
                {0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1},
                {0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1}
        };


        TimeSeriesInstance instance1 = new TimeSeriesInstance(data1,0.0);
        TimeSeriesInstance instance2 = new TimeSeriesInstance(data2,1.0);
        TimeSeriesInstance instance3 = new TimeSeriesInstance(data3,2.0);

        ShapeletDistanceEuclidean sde  = new ShapeletDistanceEuclidean();
        ShapeletIndependentMV shapelet = new ShapeletIndependentMV(1, 5, 0, 0.0,  0, instance1);
        System.out.println(sde.calculate(shapelet, instance2));
    }

}
