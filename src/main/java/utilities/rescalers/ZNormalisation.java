/* 
 * This file is part of the UEA Time Series Machine Learning (TSML) toolbox.
 *
 * The UEA TSML toolbox is free software: you can redistribute it and/or 
 * modify it under the terms of the GNU General Public License as published 
 * by the Free Software Foundation, either version 3 of the License, or 
 * (at your option) any later version.
 *
 * The UEA TSML toolbox is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along
 * with the UEA TSML toolbox. If not, see <https://www.gnu.org/licenses/>.
 */
 
package utilities.rescalers;

/**
 *
 * @author a.bostrom1
 */
public class ZNormalisation implements SeriesRescaler{

     public static final double ROUNDING_ERROR_CORRECTION = 0.000000000000001;   
    
    @Override
    public double[] rescaleSeries(double[] series) {
        return rescaleSeries(series, false);
    }

     /**
     * Z-Normalise a time series
     *
     * @param series the input time series to be z-normalised
     * @param hasClassValue specify whether the time series includes a class value
     * @return a z-normalised version of input
     */
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
        stdv = 0;
        double temp;
        for (int i = 0; i < inputLength; i++)
        {
            temp = (series[i] - mean);
            stdv += temp * temp;
        }

        stdv /= (double) inputLength;

        // if the variance is less than the error correction, just set it to 0, else calc stdv.
        stdv = (stdv < ROUNDING_ERROR_CORRECTION) ? 0.0 : Math.sqrt(stdv);
        
        //System.out.println("mean "+ mean);
        //System.out.println("stdv "+stdv);
        
        for (int i = 0; i < inputLength; i++)
        {
            //if the stdv is 0 then set to 0, else normalise.
            output[i] = (stdv == 0.0) ? 0.0 : ((series[i] - mean) / stdv);
        }

        if (hasClassValue)
        {
            output[output.length - 1] = series[series.length - 1];
        }

        return output;
    }
    
}
