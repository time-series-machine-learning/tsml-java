package tsml.classifiers.shapelet_based.dev.distances;

public class ShapeletDistanceEuclidean implements ShapeletDistanceFunction {



    @Override
    public double calculate(double[] shapelet, double[] instance) {
     /*   double bestSum = Double.MAX_VALUE;

        double sum=0;
        double[] subseq;
        double temp;
        int shapeletLength = shapelet.getLength();
        //m-l+1
        //multivariate instances that are split dont have a class value on them.
        int minLength = instance.getMinLength();

        for (int i = 0; i < minLength - shapeletLength + 1; i++)
        {
            sum = fun.sDist(i,shapelet,instance);
//            sum += fun.sDist(i,shapelet,instance);
            if (sum < bestSum)
            {
                bestSum = sum;
            }


        }
    //    sum = sum / (double)(minLength - shapeletLength + 1);

        double dist = (bestSum == 0.0) ? 0.0 : (1.0 / shapeletLength * bestSum);
  //        double dist = (sum == 0.0) ? 0.0 : ((1.0 / shapeletLength) * sum);
        return dist;*/
     return 0;
    }




}
