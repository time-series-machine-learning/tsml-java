package tsml.classifiers.shapelet_based.dev.distances;

import java.util.Arrays;

import static utilities.rescalers.ZNormalisation.ROUNDING_ERROR_CORRECTION;

public class ShapeletDistanceImprovedOnline implements ShapeletDistanceFunction {




    @Override
    public double calculate(double[] shapelet, double[] timeSeries) {
        int startPos = 0;
        DoubleWrapper sumPointer = new DoubleWrapper();
        DoubleWrapper sum2Pointer = new DoubleWrapper();
        int length = shapelet.length;
        //Generate initial subsequence that starts at the same position our candidate does.
        double[] subseq = new double[shapelet.length];
        System.arraycopy(timeSeries, startPos, subseq, 0, subseq.length);
        subseq = zNormalise(subseq, false, sumPointer, sum2Pointer);

        double bestDist = 0.0;
        double temp;
        int count = 0;

        //Compute initial distance. from the startPosition the candidate was found.
        for (int i = 0; i < length; i++)
        {
            temp = shapelet[i] - subseq[i];
            bestDist = bestDist + (temp * temp);
            count++;
        }


        int i=1;
        double currentDist;

        int[] pos = new int[2];
        double[] sum = {sumPointer.get(), sumPointer.get()};
        double[] sumsq = {sum2Pointer.get(), sum2Pointer.get()};
        boolean[] traverse = {true,true};


        while(traverse[0] || traverse[1])
        {
            //i will be 0 and 1.
            for(int j=0; j<2; j++)
            {
                int modifier = j==0 ? -1 : 1;

                pos[j] = startPos + (modifier*i);

                //if we're going left check we're greater than 0 if we're going right check we've got room to move.
                traverse[j] = j==0 ? pos[j] >= 0 : pos[j] < timeSeries.length - length;

                //if we can't traverse in that direction. skip it.
                if(!traverse[j] )
                    continue;

                //either take off nothing, or take off 1. This gives us our offset.
                double start = timeSeries[pos[j]-j];
                double end   = timeSeries[pos[j]-j + length];

                sum[j] = sum[j] + (modifier*end) - (modifier*start);
                sumsq[j] = sumsq[j] + (modifier *(end * end)) - (modifier*(start * start));

                currentDist = calculateBestDistance(shapelet, pos[j], timeSeries, bestDist, sum[j], sumsq[j]);

                if (currentDist < bestDist)
                {
                    bestDist = currentDist;
                }
            }
            i++;
        }



        bestDist = (bestDist == 0.0) ? 0.0 : (1.0 / length * bestDist);

        return bestDist;
    }

    final double[] zNormalise(double[] input, boolean classValOn, DoubleWrapper sum, DoubleWrapper sum2) {
        double mean;
        double stdv;

        double classValPenalty = classValOn ? 1 : 0;

        double[] output = new double[input.length];
        double seriesTotal = 0;
        double seriesTotal2 = 0;

        for (int i = 0; i < input.length - classValPenalty; i++) {
            seriesTotal += input[i];
            seriesTotal2 += (input[i] * input[i]);
        }

        if (sum != null && sum2 != null) {
            sum.set(seriesTotal);
            sum2.set(seriesTotal2);
        }

        mean = seriesTotal / (input.length - classValPenalty);
        double num = (seriesTotal2 - (mean * mean * (input.length - classValPenalty))) / (input.length - classValPenalty);
        stdv = (num <= ROUNDING_ERROR_CORRECTION) ? 0.0 : Math.sqrt(num);

        for (int i = 0; i < input.length - classValPenalty; i++) {
            output[i] = (stdv == 0.0) ? 0.0 : (input[i] - mean) / stdv;
        }

        if (classValOn) {
            output[output.length - 1] = input[input.length - 1];
        }

        return output;
    }

    protected double[][] sortedIndices;

    protected double calculateBestDistance(double[] shapelet, int i, double[] timeSeries, double bestDist, double sum, double sum2)
    {
        sortedIndices = sortIndexes(shapelet);
        int length = shapelet.length;
        //Compute the stats for new series
        double mean = sum / length;

        //Get rid of rounding errors
        double stdv2 = (sum2 - (mean * mean * length)) / length;

        double stdv = (stdv2 < ROUNDING_ERROR_CORRECTION) ? 0.0 : Math.sqrt(stdv2);


        //calculate the normalised distance between the series
        int j = 0;
        double currentDist = 0.0;
        double toAdd;
        int reordedIndex;
        double normalisedVal = 0.0;
        boolean dontStdv = (stdv == 0.0);

        while (j < length  && currentDist < bestDist)
        {
            //count ops
            //count++;

            reordedIndex = (int) sortedIndices[j][0];
            //if our stdv isn't done then make it 0.
            normalisedVal = dontStdv ? 0.0 : ((timeSeries[i + reordedIndex] - mean) / stdv);
            toAdd = shapelet[reordedIndex] - normalisedVal;
            currentDist = currentDist + (toAdd * toAdd);
            j++;
        }


        return currentDist;
    }

    public static double[][] sortIndexes(double[] series)
    {
        //Create an boxed array of values with corresponding indexes
        double[][] sortedSeries = new double[series.length][2];
        for (int i = 0; i < series.length; i++)
        {
            sortedSeries[i][0] = i;
            sortedSeries[i][1] = Math.abs(series[i]);
        }


        //this is the lamda expression.
        //Arrays.sort(sortedSeries, (double[] o1, double[] o2) -> Double.compare(o1[1],o2[1]));
        Arrays.sort(sortedSeries, (double[] o1, double[] o2) -> Double.compare(o2[1], o1[1]));

        return sortedSeries;
    }

    protected static class DoubleWrapper {

        private double d;

        public DoubleWrapper() {
            d = 0.0;
        }

        public DoubleWrapper(double d) {
            this.d = d;
        }

        public void set(double d) {
            this.d = d;
        }

        public double get() {
            return d;
        }
    }

}
