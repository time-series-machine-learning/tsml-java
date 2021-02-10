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
 * 
 * 
 * This class just wraps up the series rescaler for no rescaling. 
 * It allows the user to obfuscate to using classes what type of rescaling we're doing
 * as they shouldn't care.
 */
public class NoRescaling implements SeriesRescaler{

    @Override
    public double[] rescaleSeries(double[] series) {
        return rescaleSeries(series, false);
    }

    @Override
    public double[] rescaleSeries(double[] series, boolean hasClassValue) {
        return series;
    }
    
}
