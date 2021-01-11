package tsml.classifiers.shapelet_based.distances;

import tsml.classifiers.shapelet_based.classifiers.ShapeletMV;

public class ShapeletDistanceEuclidean implements ShapeletDistanceMV {
    @Override
    public double distance(ShapeletMV shapelet, double[][] instance, int seriesLength) {
        double bestSum = Double.MAX_VALUE;
        double sum;
        double[] subseq;
        double temp;
        int shapeletLength = shapelet.getLength();
        double[][] shapeletData = shapelet.getData();
        //m-l+1
        //multivariate instances that are split dont have a class value on them.
        for (int i = 0; i < seriesLength -  + 1; i++)
        {
            sum = 0;

            for(int j=0; j< instance.length; j++){

                for (int k = 0; k < shapeletLength; k++)
                {
                    //count ops
                    //incrementCount();
                    temp = shapeletData[j][k] - instance[j][k+i];
                    sum = sum + (temp * temp);
                }
            }

            if (sum < bestSum)
            {
                bestSum = sum;
                //System.out.println(i);
            }
        }

        double dist = (bestSum == 0.0) ? 0.0 : (1.0 / length * bestSum);
        return dist;
    }
}
