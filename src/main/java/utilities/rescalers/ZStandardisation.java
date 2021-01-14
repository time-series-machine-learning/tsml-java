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
