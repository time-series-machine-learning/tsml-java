package tsml.classifiers.shapelet_based.distances;

import tsml.classifiers.shapelet_based.type.ShapeletMV;

public class ShapeletDistanceEuclidean implements ShapeletDistanceMV {
    @Override
    public double calculate(ShapeletMV shapelet, double[][] instance) {
        double bestSum = Double.MAX_VALUE;
        double sum;
        double[] subseq;
        double temp;
        int shapeletLength = shapelet.getLength();
        //m-l+1
        //multivariate instances that are split dont have a class value on them.
        int minLength = Integer.MAX_VALUE;
        for(int i=0;i<instance.length;i++){
            minLength = Math.min(minLength,instance[i].length);
        }
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

    public double calculate(double[][] a, double[][] b, double shapeletLength) {

            double temp,sum=0;

            for(int i=0; i< a.length; i++){

                for (int j = 0; j < a[i].length; j++)
                {
                    temp = b[i][j] - a[i][j];
                    sum = sum + (temp * temp);
                }
            }


        double dist = (sum == 0.0) ? 0.0 : (1.0 / shapeletLength * sum);
        return dist;
    }



}
