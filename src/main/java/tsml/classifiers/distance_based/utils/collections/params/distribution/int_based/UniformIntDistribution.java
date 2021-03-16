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
 
package tsml.classifiers.distance_based.utils.collections.params.distribution.int_based;

import tsml.classifiers.distance_based.utils.collections.intervals.IntInterval;
import tsml.classifiers.distance_based.utils.collections.params.distribution.ClampedDistribution;

import java.util.Random;

/**
 * Purpose: // todo - docs - type the purpose of the code here
 * <p>
 * Contributors: goastler
 */
public class UniformIntDistribution extends ClampedDistribution<Integer> {

    public UniformIntDistribution() {
        this(0, 1);
    }

    public UniformIntDistribution(final Integer start, final Integer end) {
        super(new IntInterval(start, end));
    }
    
    public Integer sample(Random random) {
        int end = getEnd();
        int start = getStart();
        return random.nextInt(Math.abs(end - start) + 1) + Math.min(start, end);
    }
}
