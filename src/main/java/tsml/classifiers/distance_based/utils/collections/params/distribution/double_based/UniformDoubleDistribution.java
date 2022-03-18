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
 
package tsml.classifiers.distance_based.utils.collections.params.distribution.double_based;

import tsml.classifiers.distance_based.utils.collections.intervals.DoubleInterval;
import tsml.classifiers.distance_based.utils.collections.params.distribution.ClampedDistribution;

import java.util.Random;

/**
 * Purpose: // todo - docs - type the purpose of the code here
 * <p>
 * Contributors: goastler
 */
public class UniformDoubleDistribution extends ClampedDistribution<Double> {

    public UniformDoubleDistribution() {
        this(0d, 1d);
    }
    
    public UniformDoubleDistribution(final Double start, final Double end) {
        super(new DoubleInterval(start, end));
    }

    @Override
    public Double sample(Random random) {
        double start = getStart();
        double end = getEnd();
        return random.nextDouble() * Math.abs(end - start) + Math.min(start, end);
    }
}
