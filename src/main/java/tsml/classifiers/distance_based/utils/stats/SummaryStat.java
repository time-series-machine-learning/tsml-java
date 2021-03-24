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
 
package tsml.classifiers.distance_based.utils.stats;

import java.math.BigDecimal;

public class SummaryStat {
    private double firstValue = 0;
    private long count = 0;
    private BigDecimal sum = new BigDecimal(0);
    private BigDecimal sqSum = new BigDecimal(0);

    public void add(double v) {
        if(count == 0) {
            firstValue = v;
        }
        final double diff = v - firstValue;
        final double sqDiff = Math.pow(diff, 2);
        sum = sum.add(new BigDecimal(diff));
        sqSum = sqSum.add(new BigDecimal(sqDiff));
        count++;
    }

    public void remove(double v) {
        if(count <= 0) {
            throw new IllegalArgumentException();
        }
        count--;
        final double diff = v - firstValue;
        final double sqDiff = Math.pow(diff, 2);
        sum = sum.subtract(new BigDecimal(diff));
        sqSum = sqSum.subtract(new BigDecimal(sqDiff));
    }

    public double getMean() {
        return firstValue + sum.divide(new BigDecimal(count), BigDecimal.ROUND_HALF_UP).doubleValue();
    }

    public double getPopulationVariance() {
        if(count <= 0) {
            return 0;
        }
        return sqSum.subtract(sum.multiply(sum).divide(new BigDecimal(count), BigDecimal.ROUND_HALF_UP)).divide(new BigDecimal(
                count),
            BigDecimal.ROUND_HALF_UP).doubleValue();
    }

    public double getSampleVariance() {
        if(count <= 0) {
            return 0;
        }
        return sqSum.subtract(sum.multiply(sum).divide(new BigDecimal(count), BigDecimal.ROUND_HALF_UP)).divide(new BigDecimal(
                count - 1),
            BigDecimal.ROUND_HALF_UP).doubleValue();
    }

    public double getSampleStandardDeviation() {
        if(count <= 0) {
            return 0;
        }
        return Math.sqrt(getSampleVariance());
    }

    public double getPopulationStandardDeviation() {
        if(count <= 0) {
            return 0;
        }
        return Math.sqrt(getPopulationVariance());
    }

    public long getCount() {
        return count;
    }
}
