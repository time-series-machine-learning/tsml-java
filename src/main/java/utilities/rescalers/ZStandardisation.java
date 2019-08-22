/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package utilities.rescalers;

/**
 *
 * @author a.bostrom1
 */
public class ZStandardisation implements SeriesRescaler{

    @Override
    public double[] rescaleSeries(double[] series) {
        return rescaleSeries(series, false);
    }

    @Override
    public double[] rescaleSeries(double[] series, boolean hasClassValue) {
        double mean;
        double stdv;

        int classValPenalty = hasClassValue ? 1 : 0;
        int inputLength = series.length - classValPenalty;

        double[] output = new double[series.length];
        double seriesTotal = 0;
        for (int i = 0; i < inputLength; i++)
        {
            seriesTotal += series[i];
        }

        mean = seriesTotal / (double) inputLength;
        
        
        for (int i = 0; i < inputLength; i++)
        {
            //if the stdv is 0 then set to 0, else normalise.
            output[i] = series[i] - mean;
        }

        if (hasClassValue)
        {
            output[output.length - 1] = series[series.length - 1];
        }

        return output;
    }
    
}
